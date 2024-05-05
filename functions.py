import os
import sys
import math
import torch
import sklearn
import argparse
import itertools
import numpy as np
from time import time
from datetime import timedelta

from nltk import ngrams
from nltk.lm import NgramCounter

import matplotlib as mpl
import random

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from mpl_toolkits.axes_grid1 import make_axes_locatable

import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, MSELoss
from models import LabelSmoothingCrossEntropy
from models import LSTM
from models import Transformer

from torch.nn.functional import cosine_similarity
import pickle
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

from sklearn.metrics import confusion_matrix

from time import perf_counter

mpl.use("Agg")


###############################################################################
# Arguments
###############################################################################


def get_arguments():
    """Parse the arguments and check their values.

    Returns:
        argparse.ArgumentParser: Arguments.
    """
    # Create parser
    parser = argparse.ArgumentParser()

    # Misc
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--log_folder",
        type=str,
        default=None,
        help="name of the log folder (optional)",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="id of GPUs to use seperated by commas",
    )

    # Data
    parser.add_argument(
        "--data_path",
        type=str,
        help="path to the folder that contains the datasets",
    )
    parser.add_argument(
        "--train_folder",
        type=str,
        help="name of the folder that contains the training set "
        "(format: 'Name to display:folder')",
    )
    parser.add_argument(
        "--valid_id_folder",
        type=str,
        help="name of the folder that contains the in-distribution "
        "validation set (format: 'Name to display:folder')",
    )
    parser.add_argument(
        "--valid_ood_folders",
        type=str,
        help="name of the folders that contains the out-of-distribution "
        "validation sets (format: 'Name to display:folder1,"
        "Name to display:folder2,')",
    )
    parser.add_argument(
        "--test_id_folder",
        type=str,
        help="name of the folder that contains the in-distribution "
        "test set (format: 'Name to display:folder1,"
        "Name to display:folder2,')",
    )
    parser.add_argument(
        "--test_ood_folders",
        type=str,
        help="name of the folders that contains the out-of-distribution "
        "test sets (format: 'Name to display:folder1,"
        "Name to display:folder2,')",
    )
    parser.add_argument(
        "--generate_dataset",
        action="store_true",
        help="generate the dataset in the data folder",
    )
    parser.add_argument(
        "--max_sample",
        type=int,
        default=None,
        help="maximum number of sequences to load",
    )
    parser.add_argument(
        "--max_token", type=int, default=None, help="maximum sequence lengths"
    )
    parser.add_argument(
        "--n_categories",
        type=int,
        default=4,
        help="number of categories for latencies (choose 3, 5, 7, or 9, and +1 for entry events)",
    )
    parser.add_argument(
        "--multi_category",
        action="store_true",
        help="multiple category tags are available in data for latencies",
    )
    parser.add_argument(
        "--continuous_latency",
        action="store_true",
        help="event durations are not categorized and are used as is",
    )

    # Model
    parser.add_argument(
        "--model",
        type=str,
        default="transformer",
        choices=["ngram", "lstm", "transformer"],
        help="model to use",
    )
    parser.add_argument(
        "--load_model",
        type=str,
        default=None,
        help="load the model from the load_model log folder",
    )
    parser.add_argument(
        "--order", type=int, default=None, help="ngram order (value of n)"
    )
    parser.add_argument(
        "--dim_sys",
        type=int,
        default=None,
        help="embedding dimension of system call name",
    )
    parser.add_argument(
        "--dim_entry",
        type=int,
        default=None,
        help="embedding dimension of the entry or exit",
    )
    parser.add_argument(
        "--dim_ret",
        type=int,
        default=None,
        help="embedding dimension of the return value",
    )
    parser.add_argument(
        "--dim_proc",
        type=int,
        default=None,
        help="embedding dimension of process names",
    )
    parser.add_argument(
        "--dim_pid",
        type=int,
        default=None,
        help="embedding dimension of the process id",
    )
    parser.add_argument(
        "--dim_tid",
        type=int,
        default=None,
        help="embedding dimension of the thread id",
    )
    parser.add_argument(
        "--dim_time",
        type=int,
        default=None,
        help="embedding dimension of the elapsed time between events",
    )
    parser.add_argument(
        "--dim_order",
        type=int,
        default=None,
        help="embedding dimension of the ordering",
    )
    parser.add_argument(
        "--dim_f_mean",
        type=int,
        default=None,
        help="embedding dimension of the mean feature",
    )
    parser.add_argument(
        "--n_head",
        type=int,
        default=None,
        help="number of attention heads (d_k = d/h)",
    )
    parser.add_argument(
        "--n_hidden",
        type=int,
        default=None,
        help="number of hidden units of each encoder MLP",
    )
    parser.add_argument("--n_layer", type=int, default=None, help="number of layers")
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="model dropout rate (embedding & encoder)",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default=None,
        choices=["relu", "gelu", "swiglu"],
        help="activation function",
    )
    parser.add_argument(
        "--tfixup",
        action="store_true",
        help="uses T-fixup initialization and removes the layer normalization",
    )
    parser.add_argument(
        "--use_event_model",
        action="store_true",
        help="utilize the event model for OOD detection",
    )
    parser.add_argument(
        "--use_latency_model",
        action="store_true",
        help="utilize the latency model for OOD detection"
        "if both event model and latency model are selected"
        "they will be both used together",
    )
    parser.add_argument(
        "--train_event_model",
        action="store_true",
        help="train the model for next event prediction task",
    )
    parser.add_argument(
        "--train_latency_model",
        action="store_true",
        help="train the model for latency prediction task",
    )
    parser.add_argument(
        "--ordinal_latency",
        action="store_true",
        help="ordinality of latency categories is considered for model training and inference",
    )

    # Training
    parser.add_argument("--batch", type=int, default=None, help="batch size per GPU")
    parser.add_argument("--n_update", type=int, default=None, help="number of updates")
    parser.add_argument(
        "--eval",
        type=int,
        default=None,
        help="number of updates before evaluating the model " "(impact early stopping)",
    )
    parser.add_argument("--lr", type=float, default=None, help="learning rate")
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=None,
        help="increase the learning rate linearly for the first warmup_steps "
        "training steps, and decrease it thereafter proportionally to the "
        "inverse square root of the step number",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default=None,
        choices=["adam", "ranger"],
        help="Optimizer algorithm used for training the chosen model",
    )
    parser.add_argument(
        "--clip",
        type=float,
        default=None,
        help="maximum norm of the gradients",
    )
    parser.add_argument("--ls", type=float, default=None, help="label smoothing [0,1]")
    parser.add_argument(
        "--reduce_lr_patience",
        type=int,
        default=None,
        help="number of iterations before dividing the learning rate by 10 "
        "if the validation loss did not improve in the last (args.patience/2) "
        "evaluations by at least 0.001",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=None,
        help="number of iterations before early stopping",
    )
    parser.add_argument("--chk", action="store_true", help="use gradient checkpointing")
    parser.add_argument(
        "--amp", action="store_true", help="use automatic mixed-precision"
    )

    # Analysis
    parser.add_argument(
        "--dataset_stat",
        action="store_true",
        help="display data information and plot distributions",
    )
    parser.add_argument("--analysis", action="store_true", help="analyze the model")
    parser.add_argument(
        "--eval_token_pred",
        action="store_true",
        help="evaluate model on next token prediction task",
    )
    parser.add_argument(
        "--unique_thresh",
        action="store_true",
        help="use a single threshold for change/novelty detection of all OOD sets",
    )
    parser.add_argument(
        "--analyze_rootCause",
        action="store_true",
        help="analyze root cause identifier",
    )
    parser.add_argument(
        "--test_random_cases",
        action="store_true",
        help="test adaptive tracer on randomly generated traces",
    )

    args = parser.parse_args()

    # Assertions
    assert os.path.exists(
        os.path.join(args.data_path, args.train_folder.split(":")[1])
    ), f"{os.path.join(args.data_path, args.train_folder.split(':')[1])} "
    "does not exist"
    assert os.path.exists(
        os.path.join(args.data_path, args.valid_id_folder.split(":")[1])
    ), f"{os.path.join(args.data_path, args.valid_id_folder.split(':')[1])} "
    "does not exist"
    assert os.path.exists(
        os.path.join(args.data_path, args.test_id_folder.split(":")[1])
    ), f"{os.path.join(args.data_path, args.test_id_folder.split(':')[1])} "
    "does not exist"
    for f in args.valid_ood_folders.split(","):
        assert os.path.exists(
            os.path.join(args.data_path, f.split(":")[1])
        ), f"{os.path.join(args.data_path, f.split(':')[1])} does not exist"
    for f in args.test_ood_folders.split(","):
        assert os.path.exists(
            os.path.join(args.data_path, f.split(":")[1])
        ), f"{os.path.join(args.data_path, f.split(':')[1])} does not exist"

    assert (
        args.max_sample is None or args.max_sample > 0
    ), "The number of samples must be greater than 0"
    assert (
        args.max_token is None or args.max_token > 0
    ), "The number of samples must be greater than 0"
    assert (
        args.order is None or args.order > 1
    ), "The n-gram order must be greater than 1"
    assert (
        args.dim_sys is None or args.dim_sys >= 0
    ), "The embedding dimensions must be greater than or equal to 0"
    assert (
        args.dim_entry is None or args.dim_entry > 0
    ), "The embedding dimensions must be greater than or equal to 0"
    assert (
        args.dim_ret is None or args.dim_ret >= 0
    ), "The embedding dimensions must be greater than or equal to 0"
    assert (
        args.dim_proc is None or args.dim_proc >= 0
    ), "The embedding dimensions must be greater than or equal to 0"
    assert (
        args.dim_pid is None or args.dim_pid >= 0
    ), "The embedding dimensions must be greater than or equal to 0"
    assert (
        args.dim_tid is None or args.dim_tid >= 0
    ), "The embedding dimensions must be greater than or equal to 0"
    assert (
        args.dim_time is None or args.dim_time >= 0
    ), "The embedding dimensions must be greater than or equal to 0"
    assert (
        args.dim_order is None or args.dim_order >= 0
    ), "The embedding dimensions must be greater than or equal to 0"
    assert (
        args.dim_f_mean is None or args.dim_f_mean >= 0
    ), "The embedding dimensions must be greater than or equal to 0"
    assert (
        args.n_head is None or args.n_head > 0
    ), "The number of heads must be greater than 0"
    assert (
        args.n_hidden is None or args.n_hidden > 0
    ), "The number of units must be greater than 0"
    assert (
        args.n_layer is None or args.n_layer > 0
    ), "The number of layers must be greater than 0"
    assert (
        args.dropout is None or args.dropout >= 0 and args.dropout <= 1
    ), "The dropout probability must be greater than 0 and lower than 1"
    assert (
        args.batch is None or args.batch > 0
    ), "The number of sample per batch must be greater than 0"
    assert (
        args.n_update is None or args.n_update > 0
    ), "The number of updates must be greater than 0"
    assert (
        args.eval is None or args.eval > 0
    ), "The number of updates before evaluating the model must be greater "
    "than 0"
    assert (
        args.n_layer is None or args.n_layer > 0
    ), "The learning rate must be greater than 0"
    assert (
        args.warmup_steps is None or args.warmup_steps >= 0
    ), "The number of warmup steps must be greater than 0"
    assert (
        args.clip is None or args.clip > 0
    ), "The gradients' maximum norm must be greater than 0"
    assert args.ls is None or (
        args.ls >= 0 and args.ls <= 1
    ), "The label smoothing coefficient must be greater than 0 and lower "
    "than 1"
    assert (
        args.early_stopping_patience is None or args.early_stopping_patience > 0
    ), "The number of updates before early stopping must be greater than 0"
    assert (
        args.reduce_lr_patience is None or args.reduce_lr_patience > 0
    ), "The number of updates before reducing the learning rate must be "
    "greater than 0"

    if args.model != "ngram":
        assert args.gpu is not None, "Neural networks require a GPU"
        assert (
            len(args.gpu.split(",")) <= torch.cuda.device_count()
        ), f"Only {torch.cuda.device_count()} GPU available"

    if args.model == "lstm":
        if args.chk:
            args.chk = False
            print("Checkpoint is not implemented for the LSTM")
        if args.tfixup:
            args.tfixup = False
            print("T-fixup is not defined for the LSTM")
        if args.activation is not None:
            args.activation = None
            print("The activation functions are fixed for the LSTM")

    if os.path.exists(args.log_folder):
        print(f"{args.log_folder} already exists")

    return args


###############################################################################
# Data
###############################################################################


def load_trace(file_path):
    """Load the trace located in path.

    Args:
        file_path (str): Path to the LTTng trace folder.

    Returns:
        babeltrace.TraceCollection: A collection of one trace.
    """
    # Load babeltrace in the function to remove the import if the dataset
    # has already been generated (babeltrace not available on Compute Canada)
    try:
        import bt2
    except ImportError:
        raise ImportError("Library bt2 is not available (https://babeltrace.org)")

    return bt2.TraceCollectionMessageIterator(file_path)


def get_events(trace_collection, keys=None):
    """Return a generator of events. An event is a dict with the key the
    arguement's name.

    Args:
        trace_collection (babeltrace.TraceCollection): Trace from which
            to read the events.
        keys (dict, optional): Dictionary of the multiple ways of the arguments
            to consider in addition to name and elapsed time between events.

    Returns:
        generator: A generator of events.
    """
    # Load babeltrace in the function to remove the import if the dataset
    # has already been generated (babeltrace not available on Compute Canada)
    try:
        import bt2
    except ImportError:
        raise ImportError("Library bt2 is not available " "(https://babeltrace.org)")

    for msg in trace_collection:
        if type(msg) is not bt2._EventMessageConst:
            continue

        event = dict()
        event["name"] = msg.event.name
        event["timestamp"] = msg.default_clock_snapshot.ns_from_origin

        event["sector"] = None
        event["ptr"] = None
        event["next_tid"] = None
        if event["name"] == "block_rq_insert":
            event["name"] = "block_rq_entry"
            event["sector"] = msg.event["sector"]
        elif event["name"] == "block_rq_complete":
            event["name"] = "block_rq_exit"
            event["sector"] = msg.event["sector"]

        elif event["name"] == "timer_hrtimer_start":
            event["name"] = "timer_hrtimer_entry"
        elif event["name"] == "timer_hrtimer_cancel":
            event["name"] = "timer_hrtimer_exit"

        elif event["name"] == "timer_start":
            event["name"] = "timer_entry"
        elif event["name"] == "timer_cancel":
            event["name"] = "timer_exit"

        elif event["name"] == "sched_switch":
            event["next_tid"] = msg.event["next_tid"]

        if (
            event["name"] == "httpd:enter_event_handler"
            or event["name"] == "httpd:exit_event_handler"
        ):
            if msg.event.payload_field["connection_state"] is None:
                event["connection_state"] = -1
            else:
                event["connection_state"] = int(
                    msg.event.payload_field["connection_state"]
                )
        else:
            event["connection_state"] = -1

        for k, v in keys.items():
            try:
                event[v] = msg.event[k]
            except KeyError:
                continue

        yield event


def get_requests(events):
    """Split individual requests from Apache. Note that this implementation
    is not the fastest, but requires very little memory.

    Args:
        events (generator): Generator of event.

    Yields:
        list: A list of events corresponding to a request.
    """
    # Dictionary of active threads
    threads = {}
    event_time_map = {}
    event_time_map_hist = {}
    error_count = 0

    for event in events:
        latency_val, f_mean_val, error_count_val = get_duration(
            event, event_time_map, event_time_map_hist
        )
        # Start the request for a specific thread
        if event["name"] == "httpd:enter_event_handler":
            # Filter connections that lingers (not real requests)
            if event["connection_state"] not in [6, 7]:
                threads[event["tid"]] = []

        # End the request for a specific thread
        elif event["name"] == "httpd:exit_event_handler":
            if event["tid"] in threads:
                if threads[event["tid"]]:
                    yield threads[event["tid"]], event_time_map_hist, error_count
                    error_count = 0
                del threads[event["tid"]]

        # Add the system calls in all currently recording thread
        else:
            event["latency"] = latency_val
            event["f_mean"] = f_mean_val
            error_count += error_count_val
            for request in threads.values():
                request.append(event)


def get_duration(event, event_time_map, event_time_map_hist):
    sysname = event["name"].replace("syscall_", "")
    sysname = sysname.replace("entry_", "").replace("exit_", "")
    tmp_id = sysname + "@" + str(event["pid"]) + str(event["tid"])
    prev_time = event_time_map.get(tmp_id)
    current_time = event["timestamp"]
    f_mean, latency = None, None
    error_count = 0
    if "entry" in event["name"]:
        if prev_time is None:
            event_time_map.update({tmp_id: current_time})
        else:
            if prev_time == -1:
                event_time_map.update({tmp_id: current_time})
            else:
                error_count += 1
        f_mean = 0
        latency = 0
    elif "exit" in event["name"]:
        if prev_time is None:
            f_mean = 0
            latency = 0
            event_time_map.update({tmp_id: -1})
            error_count += 1
        else:
            if prev_time == -1:
                f_mean = 0
                latency = 0
                event_time_map.update({tmp_id: -1})
                error_count += 1
            else:
                event_time_map.update({tmp_id: -1})
                # remove the tid and pid so the history is about the same system call (for all tid and pid)
                tmp_id_hist = tmp_id.split("@")[0]
                tmp_hist = event_time_map_hist.get(tmp_id_hist)

                latency = current_time - prev_time
                if tmp_hist is None:
                    event_time_map_hist.update({tmp_id_hist: [latency]})
                    f_mean = 0
                else:
                    if len(tmp_hist) > 10:
                        f_mean = sum(tmp_hist[-10:]) / 10
                    else:
                        f_mean = sum(tmp_hist) / len(tmp_hist)
                    tmp_hist.append(latency)

    return latency, f_mean, error_count


def generate_dataset_request_based(
    file_path,
    dict_sys,
    dict_proc,
    n_categories,
    train=False,
    event_delay_spans=[None, None, None, None],
    add_noise=False,
):
    """Generate the dataset and write it iteratively into a file
    that will be iteratively read by the Dataloader.

    Args:
        file_path (str): Path to the file to load.
        dict_sys (dataset.Dictionary): Vocabulary of system call names.
        dict_proc (dataset.Dictionary): Vocabulary of process names.
        train (bool): Whether to update the dictionaries.
    """
    # Open the trace
    trace = load_trace(file_path)

    # Mapping to consider the multiple way of denoting each argument
    # (e.g., the tid may be stored as 'tid' or 'vtid')
    keys = {
        "vtid": "tid",
        "tid": "tid",
        "vpid": "pid",
        "pid": "pid",
        "procname": "procname",
        "ret": "ret",
    }

    start = time()

    call_arr, proc_arr, entry_arr, duration_arr, pid_arr, tid_arr, ret_arr = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    latency_arr, f_mean_arr, complete_names_arr, time_stamp_arr = [], [], [], []

    # Start a sequence with the token [START] with no argument (0s)
    call = [dict_sys.get_idx("[START]")]
    proc = [dict_proc.get_idx("[START]")]
    entry, duration, pid, tid, ret, time_stamp = [0], [0], [0], [0], [0], []
    prev_timestp = None

    event_time_map_hist = None
    latency, f_mean = [0], [0]
    event_complete_names = [""]
    error_count = 0
    save_thresh = 50000 if train else 10000

    # Open the file
    f = open(f"{file_path}/data.txt", "w")

    top_10 = [
        "recvfrom",
        "read",
        "poll",
        "sendto",
        "rt_sigaction",
        "fcntl",
        "futex",
        "writev",
        "close",
        "ppoll",
    ]

    count_ignored, count_total = 0, 0

    # Skip the first 1,000 requests as issues may occur when tracing starts.
    # Note that 1,000 requests per second were sent, so it amounts to skipping
    # the first second (which is on the cautious side)
    for i, (request, tmp_time_hist, tmp_err_count) in enumerate(
        itertools.islice(get_requests(get_events(trace, keys)), 1000, None)
    ):
        print(
            f"\rReading {file_path:40s}: {i:9d} - Error#: {error_count} "
            f"({timedelta(seconds=round(time() - start))})",
            end="",
            file=sys.stderr,
            flush=True,
        )
        for event in request:
            # Get system call and process names
            sysname = event["name"].replace("syscall_", "")
            sysname = sysname.replace("entry_", "").replace("exit_", "")
            procname = str(event["procname"])

            count_total += 1
            # if sysname in top_10:
            #     count_ignored += 1
            #     continue

            # If it is the train set
            if train:
                # Add system call name to dictionary
                dict_sys.add_word(sysname)
                # Add process name to dicitonary
                dict_proc.add_word(procname)

            # Append system call name
            call.append(dict_sys.get_idx(sysname))
            # Append entry (1), exit (2), or none (0)
            if "entry" in event["name"]:
                entry.append(1)
            elif "exit" in event["name"]:
                entry.append(2)
            else:
                entry.append(0)
            # Append elapsed time between events
            if prev_timestp is not None:
                duration.append(event["timestamp"] - prev_timestp)
            else:
                duration.append(0)
            prev_timestp = event["timestamp"]
            # mark the start time of a request
            if len(time_stamp) == 0:
                time_stamp.append(event["timestamp"])
            # Append process name
            proc.append(dict_proc.get_idx(procname))
            # Append pid
            pid.append(event["pid"])
            # Append tid
            tid.append(event["tid"])
            # Append return value
            if "entry" in event["name"]:
                ret.append(0)  # start event (no return value)
            elif event["ret"] >= 0:
                ret.append(1)  # success
            else:
                ret.append(2)  # failure

            # append latency
            latency.append(event["latency"])
            f_mean.append(event["f_mean"])
            # used for categorization of latencies
            event_complete_names.append(event["name"])

        # End the sequence with the token [END] with no argument (0s)
        call.append(dict_sys.get_idx("[END]"))
        proc.append(dict_proc.get_idx("[END]"))
        entry.append(0)
        duration.append(0)
        pid.append(0)
        tid.append(0)
        ret.append(0)
        latency.append(0)
        f_mean.append(0)
        event_complete_names.append("")

        # Append the sequence to the list of sequences
        call_arr.append(call)
        proc_arr.append(proc)
        entry_arr.append(entry)
        duration_arr.append(duration)
        time_stamp_arr.append(time_stamp)
        pid_arr.append(pid)
        tid_arr.append(tid)
        ret_arr.append(ret)
        latency_arr.append(latency)
        f_mean_arr.append(f_mean)
        complete_names_arr.append(event_complete_names)

        call = [dict_sys.get_idx("[START]")]
        proc = [dict_proc.get_idx("[START]")]
        entry, duration, pid, tid, ret, time_stamp = [0], [0], [0], [0], [0], []
        latency, f_mean = [0], [0]
        event_complete_names = [""]
        prev_timestp = None

        event_time_map_hist = tmp_time_hist
        error_count += tmp_err_count

        if (i + 1) % save_thresh == 0:
            print("\ncount ignored", count_ignored, "count total", count_total)
            # categorize system call latencies
            categorized_latencies = [None, None, None, None]
            for i in range(4):
                n_cat = i * 2 + 4  # 4, 6, 8, 10
                categorized_latencies[i], _, event_delay_spans[i] = categorize_latency(
                    latency_arr,
                    f_mean_arr,
                    event_time_map_hist,
                    complete_names_arr,
                    n_cat,
                    event_delay_spans[i],
                    train,
                )

            merged_latencies = []
            for a, b, c, d in zip(*categorized_latencies):
                tmp_req_latencies = []
                for i in range(len(a)):
                    tmp_req_latencies.append(
                        str(a[i]) + "-" + str(b[i]) + "-" + str(c[i]) + "-" + str(d[i])
                    )
                merged_latencies.append(tmp_req_latencies)

            for i in range(len(call_arr)):
                f.write(",".join(map(str, call_arr[i])) + ";")
                f.write(",".join(map(str, entry_arr[i])) + ";")
                f.write(",".join(map(str, duration_arr[i])) + ";")
                f.write(",".join(map(str, proc_arr[i])) + ";")
                f.write(",".join(map(str, pid_arr[i])) + ";")
                f.write(",".join(map(str, tid_arr[i])) + ";")
                f.write(",".join(map(str, ret_arr[i])) + ";")
                # f.write(",".join(map(str, f_mean_arr[i])) + ";")
                f.write(",".join(map(str, time_stamp_arr[i])) + ";")
                f.write(",".join(map(str, merged_latencies[i])) + ";")
                # Add the duration in ms
                f.write(str(sum(duration_arr[i]) / 1e6) + "\n")

            (
                call_arr,
                proc_arr,
                entry_arr,
                duration_arr,
                pid_arr,
                tid_arr,
                ret_arr,
                time_stamp_arr,
            ) = ([], [], [], [], [], [], [], [])
            latency_arr, f_mean_arr, complete_names_arr = [], [], []
            # only keep a portion of history of duration for each event
            for k in event_time_map_hist:
                event_time_map_hist[k] = event_time_map_hist[k][-20:]

            if save_thresh > 10000:
                save_thresh = 10000

    categorized_latencies = [None, None, None, None]
    for i in range(4):
        n_cat = i * 2 + 4  # 4, 6, 8, 10
        categorized_latencies[i], _, event_delay_spans[i] = categorize_latency(
            latency_arr,
            f_mean_arr,
            event_time_map_hist,
            complete_names_arr,
            n_cat,
            event_delay_spans[i],
            train,
        )
    merged_latencies = []
    for a, b, c, d in zip(*categorized_latencies):
        tmp_req_latencies = []
        for i in range(len(a)):
            tmp_req_latencies.append(
                str(a[i]) + "-" + str(b[i]) + "-" + str(c[i]) + "-" + str(d[i])
            )
        merged_latencies.append(tmp_req_latencies)

    for i in range(len(call_arr)):
        f.write(",".join(map(str, call_arr[i])) + ";")
        f.write(",".join(map(str, entry_arr[i])) + ";")
        f.write(",".join(map(str, duration_arr[i])) + ";")
        f.write(",".join(map(str, proc_arr[i])) + ";")
        f.write(",".join(map(str, pid_arr[i])) + ";")
        f.write(",".join(map(str, tid_arr[i])) + ";")
        f.write(",".join(map(str, ret_arr[i])) + ";")
        # f.write(",".join(map(str, f_mean_arr[i])) + ";")
        f.write(",".join(map(str, time_stamp_arr[i])) + ";")
        f.write(",".join(map(str, merged_latencies[i])) + ";")
        # Add the duration in ms
        f.write(str(sum(duration_arr[i]) / 1e6) + "\n")

    # Close the file
    f.close()

    # save the event delay spans for datasets other than train set
    return event_delay_spans


def get_duration_spans(file_path, n_categories):
    """Load a trace and return the duration spans of each event type.

    Args:
        file_path (str): Path to the file to load.
        dict_sys (dataset.Dictionary): Vocabulary of system call names.
        dict_proc (dataset.Dictionary): Vocabulary of process names.
        train (bool): Whether to update the dictionaries.
    """
    # Open the trace
    trace = load_trace(file_path)

    # Mapping to consider the multiple way of denoting each argument
    # (e.g., the tid may be stored as 'tid' or 'vtid')
    keys = {
        "vtid": "tid",
        "tid": "tid",
        "vpid": "pid",
        "pid": "pid",
        "cpu_id": "cpu_id",
        "procname": "procname",
        "ret": "ret",
    }

    start = time()

    latency_arr, f_mean_arr, complete_names_arr = [], [], []
    latency, f_mean = [0], [0]
    event_complete_names = [""]
    error_count = 0

    event_delay_spans = None
    save_thresh = 5000

    # Skip the first 1,000 requests as issues may occur when tracing starts.
    # Note that 1,000 requests per second were sent, so it amounts to skipping
    # the first second (which is on the cautious side)
    # MUST BE CHANGED TO 100 FOR THE TOY DATASETS (#TODO)
    for i, (request, tmp_time_hist, tmp_err_count, tmp_error_map) in enumerate(
        itertools.islice(get_requests(get_events(trace, keys)), 1000, None)
    ):
        print(
            f"\rReading {file_path:40s}: {i:9d} - Error#: {error_count} "
            f"({timedelta(seconds=round(time() - start))})",
            end="",
            file=sys.stderr,
            flush=True,
        )
        for event in request:
            latency.append(event["latency"])
            f_mean.append(event["f_mean"])
            # used for categorization of latencies
            event_complete_names.append(event["name"])

        latency.append(0)
        f_mean.append(0)
        event_complete_names.append("")

        latency_arr.append(latency)
        f_mean_arr.append(f_mean)
        complete_names_arr.append(event_complete_names)

        latency, f_mean = [0], [0]
        event_complete_names = [""]

        event_time_map_hist = tmp_time_hist
        error_count += tmp_err_count

        if (i + 1) % save_thresh == 0:
            # categorize system call latencies
            latency_arr, f_mean_arr, event_delay_spans = categorize_latency(
                latency_arr,
                f_mean_arr,
                event_time_map_hist,
                complete_names_arr,
                n_categories,
                event_delay_spans,
                update_spans=True,
            )

            latency_arr, f_mean_arr, complete_names_arr = [], [], []
            # only keep a portion of history of duration for each event
            for k in event_time_map_hist:
                event_time_map_hist[k] = event_time_map_hist[k][-10:]

            if i > 50000:
                break

    return event_delay_spans


def categorize_latency(
    latency,
    f_mean,
    event_time_map_hist,
    event_complete_names,
    n_categories,
    event_delay_spans=None,
    update_spans=False,
):
    # exclude the none category for upcomming calculations
    n_categories = n_categories - 1
    if update_spans:
        # find the delay spans for current subset of events
        tmp_delay_spans = {}
        for event in event_time_map_hist:
            delays_arr = np.array(event_time_map_hist.get(event))
            delays_arr = np.sort(delays_arr)

            # Calculate the arithmetic total for the given number of categories
            total = n_categories * (n_categories + 1) / 2

            # Calculate the cumulative percentage for each category boundary
            cumulative_percentage = 0
            percentiles = []

            for i in range(
                1, n_categories
            ):  # Note that we go up to n_categories - 1 because the last category takes up the remainder
                fraction = (n_categories - i + 1) / total
                cumulative_percentage += fraction * 100  # Convert to percentage
                percentiles.append(cumulative_percentage)

            # Fetch the values at those percentiles
            boundaries = np.percentile(delays_arr, percentiles)
            tmp_delay_spans.update({event: (boundaries, len(delays_arr))})

        # combine the delay spans of current subset with the delay spans of previous subsets
        if event_delay_spans is None:
            # this is the first subset
            event_delay_spans = tmp_delay_spans
        else:
            # use weighted average to update the delay spans of each event
            for event in tmp_delay_spans:
                if event in event_delay_spans:
                    prev_count = event_delay_spans.get(event)[1]
                    prev_spans = event_delay_spans.get(event)[0]
                    new_count = tmp_delay_spans.get(event)[1]
                    new_spans = tmp_delay_spans.get(event)[0]

                    weighted_average = (
                        prev_spans * prev_count + new_spans * new_count
                    ) / (prev_count + new_count)
                    event_delay_spans.update(
                        {event: (weighted_average, prev_count + new_count)}
                    )

                    # print(
                    #     "event",
                    #     event,
                    #     " change%",
                    #     (weighted_average - prev_spans) / prev_spans,
                    #     new_count,
                    #     prev_count,
                    # )
                else:
                    # this event was not in the previous subsets
                    event_delay_spans.update({event: tmp_delay_spans.get(event)})
    else:
        # use the delay spans of previous subsets
        if event_delay_spans is None:
            raise Exception("event_delay_spans is None")

    categorized_latencies = []
    categorized_mean = []
    _categorized_l = []
    _categorized_m = []
    for _latency, _mean, _events in zip(latency, f_mean, event_complete_names):
        for l, m, e in zip(_latency, _mean, _events):
            name = e.replace("entry_", "").replace("exit_", "").replace("syscall_", "")

            tmp_span_info = event_delay_spans.get(name)
            if tmp_span_info is not None:
                tmp_spans, _ = tmp_span_info
            else:
                _categorized_l.append(0)
                _categorized_m.append(0)
                continue

            def categorize_value(value, boundaries):
                """Categorize a value based on given boundaries."""
                for i, boundary in enumerate(boundaries):
                    if value <= boundary:
                        return i + 1
                return len(boundaries) + 1

            if "exit" in e:
                _categorized_l.append(categorize_value(l, tmp_spans))
                _categorized_m.append(categorize_value(m, tmp_spans))
            else:
                # we dont have latencies for entry events
                _categorized_l.append(0)
                _categorized_m.append(0)
        categorized_latencies.append(_categorized_l)
        categorized_mean.append(_categorized_m)

        _categorized_l = []
        _categorized_m = []

    return categorized_latencies, categorized_mean, event_delay_spans


def dataset_stat(file_path, dict_sys, dict_proc, name, log_folder):
    """Load a dataset from the file given in argument and print its
    statistics.

    Args:
        file_path (str): Path to the file to load.
        dict_sys (dataset.Dictionary): Vocabulary of system call names.
        dict_proc (dataset.Dictionary): Vocabulary of process names.
        name (str): Name of the dataset.
        it (int): Iteration number (log folder).
    """
    # Create directories if necessary
    if not os.path.exists(f"{log_folder}/datasets/{name}"):
        os.makedirs(f"{log_folder}/datasets/{name}")

    # Request length
    length, duration, call, proc = [], [], [], []
    with open(file_path, "r") as f:
        for line in f:
            line = line.split(";")
            length.append(len(line[0].split(",")))
            duration.append(float(line[-1]))
            call.append(list(map(int, (line[0].split(",")))))
            proc.append(list(map(int, (line[3].split(",")))))

    print("=" * 100 + f"\n{f'{name} Set':^100s}\n" + "=" * 100)
    print(f"{'Number of requests':30}: {len(length):68,}")
    print(f"{'Min requests length':30}: {min(length):68,}")
    print(
        f"{'Mean requests length':30}: "
        f"{np.mean(length):57.1f} ± {np.std(length):8.1f}"
    )
    print(f"{'Max requests length':30}: {max(length):68,}")
    print(f"{'Min request duration':30}: {min(duration):66.2f}ms")
    print(
        f"{'Mean request duration':30}: "
        f"{np.mean(duration):57.2f} ± {np.std(duration):6.2f}ms"
    )
    print(f"{'Max request duration':30}: {max(duration):66.2f}ms")

    # Plot the duration distribution and syscall/process names histograms
    plot_duration(duration, name, log_folder)
    plot_length(length, name, log_folder)
    plot_hist(call, dict_sys.idx2word, name, "System Call", log_folder)
    plot_hist(proc, dict_proc.idx2word, name, "Process", log_folder)


def collate_fn(data):
    """Construct a batch by padding the sequence.
    Args:
        data (tuple): Tensors to pad.
    Returns:
        tuple: Padded tensors.
    """
    data = list(zip(*data))
    data, time_stamp, req_duration = data[:-2], data[-2], data[-1]
    size = list(map(len, data[0]))
    pad_data = [torch.zeros(len(size), max(size), dtype=torch.int64) for _ in data]
    for i, args in enumerate(data):
        for j, sample in enumerate(args):
            pad_data[i][j][: size[j]] = torch.tensor(sample)
    pad_data = [args.type(torch.int64) for args in pad_data]
    pad_mask = (pad_data[0] == 0).type(torch.bool)
    return pad_data, pad_mask, time_stamp, req_duration


###############################################################################
# n-gram
###############################################################################


# Add n-1 padding at the start and end. Since there is already one START and
# END token, only add n-2 padding (2 = token START) and end (3 = token END)
def nltk_ngram(file_path, n, max_sample):
    """Extract n-grams from the data in the file given in parameter.

    Args:
        file_path (str): File path of the dataset
        n (int): Order of the n-gram.

    Returns:
        nltk.lm.NgramCounter: NLTK n-gram counter.
    """
    start = time()
    with open(file_path, "r") as f:
        counter = NgramCounter(
            (
                ngrams(
                    [2] * (n - 2) + list(map(int, line.split(";")[0].split(",")))[:-1],
                    n,
                )
                for line in itertools.islice(f, max_sample)
            )
        )
    print(
        f"{n}-grams extraction done in " f"{timedelta(seconds=round(time() - start))}"
    )
    return counter


def ngram_eval(file_path, counter, n, name, max_sample):
    """Evaluate the n-gram.

    Args:
        file_path (str): File path of the dataset
        counter (nltk.lm.NgramCounter): NLTK n-gram counter
        n (int): Order of the n-gram
        name (string, optional): Name of the dataset. Defaults to None.
    """
    correct, total = 0, 0
    with open(file_path, "r") as f:
        for line in itertools.islice(f, max_sample):
            seq = [2] * (n - 2) + list(map(int, line.split(";")[0].split(",")))[:-1]
            pred = [
                seq[i + n - 1] == counter[tuple(seq[i : i + n - 1])].most_common()[0][0]
                if counter[tuple(seq[i : i + n - 1])]
                else False
                for i in range(len(seq) - n + 1)
            ]
            total += len(pred)
            correct += sum(pred)
    print(f"{name:30}: {f'acc {correct / total:.1%}':>68}")


def ood_detection_ngram(
    counter,
    n,
    epsilon,
    path_valid_id,
    paths_valid_ood,
    val_ood_to_test,
    path_test_id,
    paths_test_ood,
    log_folder,
    max_sample,
):

    # Create directories if necessary
    if not os.path.exists(f"{log_folder}/evaluation/ood"):
        os.makedirs(f"{log_folder}/evaluation/ood")

    # Validation ID
    ppl_id = []
    with open(path_valid_id[1], "r") as f:
        for line in itertools.islice(f, max_sample):
            seq = [2] * (n - 2) + list(map(int, line.split(";")[0].split(",")))[:-1]
            log_likelihood = sum(
                math.log(
                    counter[tuple(seq[i : i + n - 1])][seq[i + n - 1]]
                    / counter[tuple(seq[i : i + n - 1])].N()
                    if (
                        counter[tuple(seq[i : i + n - 1])]
                        and counter[tuple(seq[i : i + n - 1])][seq[i + n - 1]] > 0
                    )
                    else epsilon
                )
                for i in range(len(seq) - n + 1)
            )
            ppl_id.append(math.exp(-log_likelihood / (len(seq) - n + 1)))

    # Test ID
    ppl_id_test = []
    with open(path_test_id[1], "r") as f:
        for line in itertools.islice(f, max_sample):
            seq = [2] * (n - 2) + list(map(int, line.split(";")[0].split(",")))[:-1]
            log_likelihood = sum(
                math.log(
                    counter[tuple(seq[i : i + n - 1])][seq[i + n - 1]]
                    / counter[tuple(seq[i : i + n - 1])].N()
                    if (
                        counter[tuple(seq[i : i + n - 1])]
                        and counter[tuple(seq[i : i + n - 1])][seq[i + n - 1]] > 0
                    )
                    else epsilon
                )
                for i in range(len(seq) - n + 1)
            )
            ppl_id_test.append(math.exp(-log_likelihood / (len(seq) - n + 1)))

    for name, path in paths_valid_ood.items():

        # FOR VALIDATION SET:
        ppl_ood = []
        with open(path, "r") as f:
            for line in itertools.islice(f, max_sample):
                seq = [2] * (n - 2) + list(map(int, line.split(";")[0].split(",")))[:-1]
                log_likelihood = sum(
                    math.log(
                        counter[tuple(seq[i : i + n - 1])][seq[i + n - 1]]
                        / counter[tuple(seq[i : i + n - 1])].N()
                        if (
                            counter[tuple(seq[i : i + n - 1])]
                            and counter[tuple(seq[i : i + n - 1])][seq[i + n - 1]] > 0
                        )
                        else epsilon
                    )
                    for i in range(len(seq) - n + 1)
                )
                ppl_ood.append(math.exp(-log_likelihood / (len(seq) - n + 1)))

        m_len = min(len(ppl_id), len(ppl_ood))
        ppl = ppl_id[:m_len] + ppl_ood[:m_len]
        y_true = [0] * m_len + [1] * m_len

        accuracy, precision, recall, fscore = [], [], [], []
        thresholds = np.arange(
            min(ppl),
            max(ppl),
            step=(max(ppl) - min(ppl)) / 100,
        )

        for t in thresholds:
            y_pred = [1 if p > t else 0 for p in ppl]
            precision.append(sklearn.metrics.precision_score(y_true, y_pred))
            recall.append(sklearn.metrics.recall_score(y_true, y_pred))
            accuracy.append(sklearn.metrics.accuracy_score(y_true, y_pred))
            fscore.append(sklearn.metrics.f1_score(y_true, y_pred))

        # Get best score based on the validation set
        # and use it to evaluate the test set
        id_best_threshold = np.argmax(fscore)
        best_threhold_value = thresholds[id_best_threshold]

        ############
        # Log
        ############
        auroc = sklearn.metrics.roc_auc_score(y_true, ppl)
        print(f"{name}:")
        print(f"{'    AUROC':30}: {auroc:68.2%}")
        print(f"{'    Recall':30}: {recall[id_best_threshold]:68.2%}")
        print(f"{'    Precision':30}: {precision[id_best_threshold]:68.2%}")
        print(f"{'    F-score':30}: {np.max(fscore):68.2%}")
        print(f"{'    Accuracy':30}: {accuracy[id_best_threshold]:68.2%}")

        # FOR TEST SET:
        # Get respective test set and its path
        name_test = val_ood_to_test[name]
        path_test = paths_test_ood[val_ood_to_test[name]]

        ppl_ood_test = []
        with open(path_test, "r") as f:
            for line in itertools.islice(f, max_sample):
                seq = [2] * (n - 2) + list(map(int, line.split(";")[0].split(",")))[:-1]
                log_likelihood = sum(
                    math.log(
                        counter[tuple(seq[i : i + n - 1])][seq[i + n - 1]]
                        / counter[tuple(seq[i : i + n - 1])].N()
                        if (
                            counter[tuple(seq[i : i + n - 1])]
                            and counter[tuple(seq[i : i + n - 1])][seq[i + n - 1]] > 0
                        )
                        else epsilon
                    )
                    for i in range(len(seq) - n + 1)
                )
                ppl_ood_test.append(math.exp(-log_likelihood / (len(seq) - n + 1)))

        m_len = min(len(ppl_id_test), len(ppl_ood_test))
        ppl_test = ppl_id_test[:m_len] + ppl_ood_test[:m_len]
        y_true = [0] * m_len + [1] * m_len
        y_pred = [1 if p > best_threhold_value else 0 for p in ppl_test]

        ############
        # Log
        ############
        precision = sklearn.metrics.precision_score(y_true, y_pred)
        recall = sklearn.metrics.recall_score(y_true, y_pred)
        accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
        fscore = sklearn.metrics.f1_score(y_true, y_pred)
        auroc = sklearn.metrics.roc_auc_score(y_true, ppl_test)

        print(f"{name_test}:")
        print(f"{'    AUROC':30}: {auroc:68.2%}")
        print(f"{'    Recall':30}: {recall:68.2%}")
        print(f"{'    Precision':30}: {precision:68.2%}")
        print(f"{'    F-score':30}: {fscore:68.2%}")
        print(f"{'    Accuracy':30}: {accuracy:68.2%}")


###############################################################################
# Train & evaluate the model
###############################################################################


def train(
    rank,
    model,
    n_syscall,
    n_category,
    n_process,
    n_head,
    n_hidden,
    n_layer,
    dropout,
    dim_sys,
    dim_entry,
    dim_ret,
    dim_proc,
    dim_pid,
    dim_tid,
    dim_order,
    dim_time,
    dim_f_mean,
    activation,
    tfixup,
    train_dataset,
    valid_dataset,
    n_update,
    reduce_lr_patience,
    early_stopping_patience,
    warmup_steps,
    lr,
    ls,
    clip,
    eval,
    batch,
    gpu,
    chk,
    amp,
    log_folder,
    train_event_model,
    train_latency_model,
    ordinal_latency,
    continuous_latency,
):
    """Create the dataloaders, build the model, and train it using
    DistributedDataParallel.

    Args:
        rank (int): Index of the GPU (mp.spawn).
        model (str): Model type.
        n_syscall (int): number of distinct system call names.
        n_category (int): number of distinct latency values.
        n_process (int): number of distinct process names.
        n_head (int): Number of heads.
        n_hidden (int): Number of hidden units.
        n_layer ([type]): Number of layers.
        dropout ([type]): Probability of dropout.
        dim_sys (int): Dimension of the system call name embedding.
        dim_entry (int): Dimension of the entry/exit embedding.
        dim_ret (int): Dimension of the return value embedding.
        dim_proc (int): Dimension of the process name embedding.
        dim_pid (int): Dimension of the PID encoding.
        dim_tid (int): Dimension of the TID encoding.
        dim_order (int): Dimension of the order encoding.
        dim_time (int): Dimension of the encoding of the elapsed time
        between events encoding.
        activation (str): Activation function
        train_dataset (torch.utils.data.IterableDataset): Training set
        valid_dataset (torch.utils.data.IterableDataset): validation set
        n_update (int): Maximum number of updates.
        reduce_lr_patience (int): Number of iterations without improvements
        before decreasing the learning rate.
        early_stopping_patience (int): Number of iterations without
        improvements before stopping the training.
        warmup_steps (int): Number of updates before the learning reaches
        its peak.
        lr (float): Learning rate.
        ls (float): Label smoothing coefficient.
        clip (float): Maximum gradient norm
        eval (int): Number of updates between two evaluations.
        batch (int): Batch size.
        gpu (int): List of GPU id
        chk (bool): Gradient checkpointing.
        it (int): Iteration number (log folder).
    """
    world_size = len(gpu)
    device = gpu[rank]

    # Initialize the process
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )

    # Log into a file
    sys.stdout = open(f"{log_folder}/log.txt", "a")

    # Create model
    if model == "lstm":
        model = LSTM(
            n_syscall,
            n_category,
            n_process,
            n_hidden,
            n_layer,
            dropout,
            dim_sys,
            dim_entry,
            dim_ret,
            dim_proc,
            dim_pid,
            dim_tid,
            dim_order,
            dim_time,
            dim_f_mean,
            train_event_model,
            train_latency_model,
            ordinal_latency,
            continuous_latency,
        ).to(device)

    if model == "transformer":
        model = Transformer(
            n_syscall,
            n_category,
            n_process,
            n_head,
            n_hidden,
            n_layer,
            dropout,
            dim_sys,
            dim_entry,
            dim_ret,
            dim_proc,
            dim_pid,
            dim_tid,
            dim_order,
            dim_time,
            dim_f_mean,
            activation,
            tfixup,
            train_event_model,
            train_latency_model,
            ordinal_latency,
        ).to(device)

    # Move the model to GPU with id rank
    model = DDP(model, device_ids=[device], find_unused_parameters=False)

    # Loss
    criterion = LabelSmoothingCrossEntropy(label_smoothing=ls)
    criterion_latency = (
        MSELoss()
        if continuous_latency
        else (
            BCEWithLogitsLoss() if ordinal_latency else CrossEntropyLoss(ignore_index=0)
        )
    )

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Initialize variables
    update_since_best, _val_loss = 0, 0
    _val_loss_latency = 0
    epoch, best_val_loss, update = 0, sys.maxsize, 1
    train_loss, val_loss, train_acc, val_acc, grad_norm = [], [], [], [], []
    (
        train_loss_latency,
        val_loss_latency,
        train_acc_latency,
        val_acc_latency,
        train_mae_latency,
        val_mae_latency,
    ) = ([], [], [], [], [], [])
    total_train_loss, total_train_pred, total_train_correct = 0, 0, 0
    (
        total_train_loss_latency,
        total_train_pred_latency,
        total_train_correct_latency,
        total_train_mae_latency,
    ) = (0, 0, 0, 0)

    # Dataloader
    train_dataset.rank = rank
    train_dataset.world_size = world_size
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch,
        collate_fn=collate_fn,
        num_workers=8,
        pin_memory=True,
    )

    # Gradient Scaler for mixed-precision
    scaler = GradScaler(enabled=amp)

    if rank == 0:
        # Log the model's information
        print("=" * 100 + f"\n{'Model':^100s}\n" + "=" * 100)
        params = filter(lambda p: p.requires_grad, model.parameters())
        n_params = sum([np.prod(p.size()) for p in params])
        print(f"{'Number of parameters':30}: {n_params:68,}")

        # Log the device(s) and gradient checkpointing
        for i in gpu:
            dname = torch.cuda.get_device_name(f"cuda:{i}")
            print(f"{'Device':30}: {dname:>68}" if i == 0 else f"{dname:>100}")

        print(
            f"{'Gradient Checkpointing':30}: " f"{'Enabled' if chk else 'Disabled':>68}"
        )
        print(f"{'Mixed-Precision':30}: " f"{'Enabled' if amp else 'Disabled':>68}")

        print("=" * 100 + f"\n{'Training':^100s}\n" + "=" * 100)

    min_vals = torch.full((n_syscall,), float("inf"), device=device, dtype=torch.float)
    max_vals = torch.full((n_syscall,), float("-inf"), device=device, dtype=torch.float)
    # min_max_vals = torch.load(f"{log_folder}/min_max_vals.pt")
    # min_vals = min_max_vals['min_vals'].to(device)
    # max_vals = min_max_vals['max_vals'].to(device)

    start, train_time = time(), time()

    while update < n_update:

        # Increment the epoch counter
        epoch += 1

        for data, pad_mask, _, _ in train_loader:

            # Increment the update counter
            update += 1

            # Stop training after n_update
            if update > n_update:
                break

            # Compute and update the learning rate
            if warmup_steps is not None and warmup_steps > 0:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr * min(
                        math.pow(update, -0.5) * math.pow(warmup_steps, 0.5),
                        update * math.pow(warmup_steps, -1),
                    )

            # Reset gradient
            optimizer.zero_grad()

            # Send tensors to device
            X = [x.to(device) for x in data[:-2]]
            y = data[-1].to(device)
            y_latency = data[-2].to(device)
            pad_mask = pad_mask.type(torch.bool).to(device)

            with autocast(enabled=amp):
                # Get prediction
                out, out_latency = model(*X, pad_mask, chk)

                # Mask the padding for the LabelSmoothingCrossEntropy
                if train_event_model:
                    y = torch.masked_select(y, ~pad_mask).reshape(-1)
                    out = torch.masked_select(out, ~pad_mask.unsqueeze(-1)).reshape(
                        -1, n_syscall
                    )
                if train_latency_model:
                    if ordinal_latency:
                        z_mask = data[-2] != 0
                        merged_mask = (~pad_mask & z_mask.to(device)).unsqueeze(-1)
                        y_latency = transform_ordinal(y_latency, n_category)
                        y_latency = torch.masked_select(y_latency, merged_mask).reshape(
                            -1, n_category - 2
                        )
                        out_latency = torch.masked_select(
                            out_latency, merged_mask
                        ).reshape(-1, n_category - 2)
                    elif continuous_latency:
                        y_latency = torch.log(y_latency + 1e-6)
                        # for i in torch.unique(X[0]):
                        #     # Mask for selecting current event
                        #     mask = X[0] == i
                        #
                        #     # Update min and max values
                        #     current_min = y_latency[mask].min()
                        #     current_max = y_latency[mask].max()
                        #
                        #     min_vals[i] = torch.min(min_vals[i], current_min)
                        #     max_vals[i] = torch.max(max_vals[i], current_max)
                        # torch.save({'min_vals': min_vals, 'max_vals': max_vals}, f"{log_folder}/min_max_vals.pt")

                        y_latency = (y_latency - min_vals[X[0]]) / (
                            max_vals[X[0]] - min_vals[X[0]]
                        )
                        # Handling cases where min and max are the same
                        y_latency[min_vals[X[0]] == max_vals[X[0]]] = 0
                        y_latency = y_latency.reshape(-1)
                        out_latency = out_latency.reshape(-1)
                    else:
                        y_latency = y_latency.reshape(-1)
                        out_latency = out_latency.reshape(-1, n_category)

                # Compute loss
                loss = criterion(out, y) if train_event_model else 0.0
                loss_latency = (
                    criterion_latency(out_latency, y_latency)
                    if train_latency_model
                    else 0.0
                )
                loss = loss + loss_latency

            # Scales loss. Calls backward() on scaled loss to create
            # scaled gradients.
            scaler.scale(loss).backward()

            # Unscales the gradients of optimizer's assigned params in-place
            scaler.unscale_(optimizer)

            if clip is not None:
                # Since the gradients of optimizer's assigned params are
                # unscaled, clips as usual
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

            # Optimizer's gradients are already unscaled, so scaler.step does
            # not unscale them, although it still skips optimizer.step() if
            # the gradients contain infs or NaNs.
            scaler.step(optimizer)

            # Updates the scale for next iteration.
            scaler.update()

            # Collect the metrics for event model
            total_train_loss += float(loss.item()) if train_event_model else 0.0
            total_train_pred += float(y.size(0))
            total_train_correct += (
                float((torch.max(out, dim=-1)[1] == y).sum().item())
                if train_event_model
                else 0.0
            )
            # Collect the metrics for duration model
            total_train_loss_latency += (
                float(loss_latency.item()) if train_latency_model else 0.0
            )
            total_train_pred_latency += float(y_latency.size(0))
            if ordinal_latency:
                predicted_labels = (torch.sigmoid(out_latency) > 0.5).sum(dim=-1)
                y_latency = y_latency.sum(dim=-1)

                total_train_correct_latency += (
                    float((predicted_labels == y_latency).sum().item())
                    if train_latency_model
                    else 0.0
                )

                total_train_mae_latency += (
                    float((torch.abs(predicted_labels - y_latency)).sum().item())
                    if train_latency_model
                    else 0.0
                )
            else:
                total_train_correct_latency += (
                    float((torch.max(out_latency, dim=-1)[1] == y_latency).sum().item())
                    if (train_latency_model and not continuous_latency)
                    else 0.0
                )

            # Collect the gradient magnitude
            if rank == 0:
                grad_norm.append(
                    sum(p.grad.data.norm(2).item() ** 2 for p in model.parameters())
                    ** 0.5
                )

            # Every eval updates, evaluate and collect metrics
            if update % eval == 0 and rank == 0:

                # Get average duration per batch in ms
                avg_d = (time() - start) * 1000 / eval

                # Evaluate model
                (
                    _val_loss,
                    _val_acc,
                    _val_loss_latency,
                    _val_acc_latency,
                    _val_mae_latency,
                    _time_eval,
                ) = evaluate(
                    model,
                    valid_dataset,
                    batch,
                    criterion,
                    criterion_latency,
                    n_syscall,
                    n_category,
                    gpu[0],
                    train_event_model,
                    train_latency_model,
                    ordinal_latency,
                    continuous_latency,
                    min_vals,
                    max_vals,
                )

                # Append metric event model
                train_loss.append(total_train_loss / eval)
                train_acc.append(total_train_correct / total_train_pred)
                val_loss.append(_val_loss)
                val_acc.append(_val_acc)
                # Append metric duration model
                train_loss_latency.append(total_train_loss_latency / eval)
                train_acc_latency.append(
                    total_train_correct_latency / total_train_pred_latency
                )
                val_loss_latency.append(_val_loss_latency)
                val_acc_latency.append(_val_acc_latency)
                if ordinal_latency:
                    train_mae_latency.append(
                        total_train_mae_latency / total_train_pred_latency
                    )
                    val_mae_latency.append(_val_mae_latency)
                else:
                    train_mae_latency.append(0)
                    val_mae_latency.append(0)

                # Save the metric for later comparison
                np.savez(
                    f"{log_folder}/training/eval.npz",
                    train_loss=train_loss,
                    val_loss=val_loss,
                    train_acc=train_acc,
                    val_acc=val_acc,
                )

                total_val_loss = _val_loss + _val_loss_latency
                # Save the model if the validation loss is the best so far
                if total_val_loss < best_val_loss - 0.001:
                    with open(f"{log_folder}/model", "wb") as f:
                        torch.save(model.module.state_dict(), f)
                        best_val_loss = total_val_loss
                        update_since_best = 0
                else:
                    update_since_best += 1

                # Display summary of the update
                print(
                    f"Updates {update:8d} "
                    f"epoch {epoch:2d} "
                    f"loss {train_loss[-1]:5.3f} "
                    f"val_loss {val_loss[-1]:5.3f} "
                    f"acc {train_acc[-1]:5.1%} "
                    f"val_acc {val_acc[-1]:5.1%} "
                    f"loss_latency {train_loss_latency[-1]:5.3f} "
                    f"val_loss_latency {val_loss_latency[-1]:5.3f} "
                    f"acc_latency {train_acc_latency[-1]:5.1%} "
                    f"val_acc_latency {val_acc_latency[-1]:5.1%} "
                    f"MAE_latency {train_mae_latency[-1]:5.1f} "
                    f"val_MAE_latency {val_mae_latency[-1]:5.1f} "
                    f"optimization @ {avg_d:3.0f}ms/batch "
                    f"inference @ {_time_eval:3.0f}ms/batch "
                    f"lr {optimizer.param_groups[0]['lr']:3.2e} "
                    f"peak_mem {torch.cuda.max_memory_allocated(0) / 1e6:5.0f}Mo"
                )
                torch.cuda.reset_peak_memory_stats(device)

                # Plot the accuracy, the loss, and the gradient norm
                plot_accuracy(train_acc, val_acc, eval, log_folder)
                plot_loss(train_loss, val_loss, eval, log_folder)
                plot_grad_norm(grad_norm, clip, log_folder)

                # Prepare to resume training
                model.train()
                total_train_loss = 0
                total_train_pred = 0
                total_train_correct = 0
                total_train_loss_latency = 0
                total_train_pred_latency = 0
                total_train_correct_latency = 0
                total_train_mae_latency = 0

                start = time()

            if update % eval == 0:
                # Only Tensors on GPU can be broadcasted
                update_since_best = torch.Tensor([update_since_best]).to(device)
                _val_loss = torch.Tensor([_val_loss]).to(device)

                # Broadcast from GPU
                dist.broadcast(update_since_best, gpu[0])
                dist.broadcast(_val_loss, gpu[0])

                # Divide the learning rate by 10 if no improvements for
                # at least reduce_lr_patience evaluation steps
                if (
                    reduce_lr_patience is not None
                    and update_since_best > reduce_lr_patience
                ):
                    lr *= 0.1

            # Early stopping
            if (
                early_stopping_patience is not None
                and update_since_best == early_stopping_patience
            ):
                break
        if (
            early_stopping_patience is not None
            and update_since_best == early_stopping_patience
        ):
            break

    if rank == 0:
        print(f"Training done in {timedelta(seconds=round(time() - train_time))}")

    # Destroy the process group
    dist.destroy_process_group(dist.group.WORLD)


def transform_ordinal(labels, n_category):
    device = labels.device

    # Create a range tensor [2, 3, ..., n_category-1] for thresholds
    thresholds = torch.arange(2, n_category, device=device).unsqueeze(0).unsqueeze(0)

    # Expand labels to [batch_size, length, n_category-2]
    # n_category includes 0 but we do not consider 0 here
    expanded_labels = labels.unsqueeze(-1).expand(-1, -1, n_category - 2)

    # Compare expanded labels with thresholds
    ordinal_labels = (expanded_labels >= thresholds).type(torch.float16)

    return ordinal_labels


def evaluate(
    model,
    dataset,
    batch,
    criterion,
    criterion_latency,
    n_syscall,
    n_category,
    device,
    train_event_model,
    train_latency_model,
    ordinal_latency,
    continuous_latency,
    min_vals=None,
    max_vals=None,
):
    """Evaluate the model on the loader using the criterion.

    Args:
        model (torch.nn.Module): Network to evaluate.
        dataset (torch.utils.data.IterableDataset): Iterable Dataset.
        batch (int): Batch size.
        criterion (torch.nn): Loss function.
        criterion_latency (torch.nn): Loss function for the latency modeling.
        n_syscall (int): Vocabulary size.
        n_category (int): Number of latency classes.
        train_event_model (bool): Whether to train the event model.
        train_latency_model (bool): Whether to train the latency model.
        ordinal_latency (bool): Whether to consider ordinality of durations

    Returns:
        tuple: The evaluation loss and accuracy.
    """
    # Dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch,
        collate_fn=collate_fn,
        num_workers=8,
        pin_memory=True,
    )

    # Evaluate model
    model.eval()
    total_val_loss, total_val_pred, total_val_correct, time_per_batch = (
        0,
        0,
        0,
        0,
    )
    (
        total_val_loss_latency,
        total_val_pred_latency,
        total_val_correct_latency,
        total_val_mae_latency,
    ) = (
        0,
        0,
        0,
        0,
    )
    predicted_durations, true_durations = np.array([]), np.array([])
    with torch.no_grad():
        for i, (data, pad_mask, _, _) in enumerate(dataloader, 1):
            # Start timer
            start = time()

            # Send tensors to device
            X = [x.to(device) for x in data[:-2]]
            y = data[-1].to(device)
            y_latency = data[-2].to(device)
            pad_mask = pad_mask.to(device)

            # Get prediction
            out, out_latency = model(*X, pad_mask, chk=False)

            # Mask the padding for the LabelSmoothingCrossEntropy
            if train_event_model:
                y = torch.masked_select(y, ~pad_mask).reshape(-1)
                out = torch.masked_select(out, ~pad_mask.unsqueeze(-1)).reshape(
                    -1, n_syscall
                )
            if train_latency_model:
                if ordinal_latency:
                    z_mask = data[-2] != 0
                    merged_mask = (~pad_mask & z_mask.to(device)).unsqueeze(-1)
                    y_latency = transform_ordinal(y_latency, n_category)
                    y_latency = torch.masked_select(y_latency, merged_mask).reshape(
                        -1, n_category - 2
                    )
                    out_latency = torch.masked_select(out_latency, merged_mask).reshape(
                        -1, n_category - 2
                    )
                elif continuous_latency:
                    y_latency = torch.log(y_latency + 1e-6)
                    for i in range(n_syscall):
                        # Mask for selecting current event
                        mask = X[0] == i
                        if mask.any():
                            # Update min and max values
                            current_min = y_latency[mask].min()
                            current_max = y_latency[mask].max()
                            min_vals[i] = torch.min(min_vals[i], current_min)
                            max_vals[i] = torch.max(max_vals[i], current_max)
                    y_latency = (y_latency - min_vals[X[0]]) / (
                        max_vals[X[0]] - min_vals[X[0]]
                    )
                    # Handling cases where min and max are the same
                    y_latency[min_vals[X[0]] == max_vals[X[0]]] = 0
                    y_latency = y_latency.reshape(-1)
                    out_latency = out_latency.reshape(-1)
                else:
                    y_latency = y_latency.reshape(-1)
                    out_latency = out_latency.reshape(-1, n_category)

            # Compute loss
            loss = criterion(out, y) if train_event_model else 0.0
            loss_latency = (
                criterion_latency(out_latency, y_latency)
                if train_latency_model
                else 0.0
            )
            loss = loss + loss_latency

            # Stop timer
            time_per_batch += time() - start

            # Collect metric for event model
            total_val_loss += float(loss.item()) if train_event_model else 0.0
            total_val_pred += float(y.size(0))
            total_val_correct += (
                float((torch.max(out, dim=-1)[1] == y).sum().item())
                if train_event_model
                else 0.0
            )
            # Collect metric for duration model
            total_val_loss_latency += (
                float(loss_latency.item()) if train_latency_model else 0.0
            )
            if ordinal_latency:
                total_val_pred_latency += float(y_latency.size(0))

                # sum = 0 corresponds to label 1, so +1 to get the correct
                # label to maintain the functioning of "analyze_latency"
                predicted_labels = (torch.sigmoid(out_latency) > 0.5).sum(dim=-1) + 1
                y_latency = y_latency.sum(dim=-1) + 1

                total_val_correct_latency += (
                    float((predicted_labels == y_latency).sum().item())
                    if train_latency_model
                    else 0.0
                )

                total_val_mae_latency += (
                    float((torch.abs(predicted_labels - y_latency)).sum().item())
                    if train_latency_model
                    else 0.0
                )
            else:
                total_val_pred_latency += float(torch.nonzero(y_latency).size(0))
                total_val_correct_latency += (
                    correct(out_latency, y_latency, n_category)
                    if (train_latency_model and not continuous_latency)
                    else 0.0
                )

            if train_latency_model and i < 1000 and not continuous_latency:
                if ordinal_latency:
                    predicted_durations = np.concatenate(
                        (predicted_durations, predicted_labels.cpu().numpy())
                    )
                    true_durations = np.concatenate(
                        (true_durations, y_latency.cpu().numpy())
                    )
                else:
                    predicted_durations = np.concatenate(
                        (
                            predicted_durations,
                            torch.argmax(out_latency.reshape(-1, n_category), dim=1)
                            .cpu()
                            .numpy(),
                        )
                    )
                    true_durations = np.concatenate(
                        (true_durations, y_latency.reshape(-1).cpu().numpy())
                    )

    if train_latency_model and not continuous_latency:
        analyze_latency(predicted_durations, true_durations, n_category)

    return (
        total_val_loss / i,
        total_val_correct / total_val_pred,
        total_val_loss_latency / i,
        total_val_correct_latency / total_val_pred_latency,
        total_val_mae_latency / total_val_pred_latency,
        time_per_batch * 1000 / i,
    )


# https://github.com/bearpaw/pytorch-classification/blob/cc9106d598ff1fe375cc030873ceacfea0499d77/utils/eval.py
def correct(output, target, tokens):
    """Computes the number of correct predictions.

    Args:
        output (torch.tensor): output of the model
        target (torch.tensor): masked labels
        tokens (int): vocabulary size

    Returns:
        int: number of correct predictions
    """
    with torch.no_grad():
        mask = target.type(torch.bool)
        labels = torch.masked_select(target, mask)
        mask = mask.unsqueeze(-1).expand_as(output)
        output = torch.masked_select(output, mask).reshape(-1, tokens)
        _, predicted = torch.max(output, dim=-1)
    return (predicted == labels).sum().item()


def analyze_latency(predictions, labels, n_latencies):
    itos_dly = {i: str(i) for i in range(n_latencies)}
    count_total, count_none, count_true, count_true_none = 0, 0, 0, 0
    tp_map, fp_map, tn_map, fn_map, total_count_map = {}, {}, {}, {}, {}

    for pred, true in zip(predictions, labels):
        count_total += 1
        true_name = itos_dly[true]
        pred_name = itos_dly[pred]

        # 0 tag corresponds to none
        if true_name == "0":
            count_none += 1
            if pred == true:
                count_true_none += 1
            continue
        if pred == true:
            count_true += 1
            tp_map[true_name] = tp_map.get(true_name, 0) + 1
            for tn_name in itos_dly.values():
                if tn_name != true_name:
                    tn_map[tn_name] = tn_map.get(tn_name, 0) + 1
        else:
            fn_map[true_name] = fn_map.get(true_name, 0) + 1
            fp_map[pred_name] = fp_map.get(pred_name, 0) + 1

        total_count_map[true_name] = total_count_map.get(true_name, 0) + 1

    for i in total_count_map.keys():
        tp, fp, tn, fn = (
            tp_map.get(i, 0),
            fp_map.get(i, 0),
            tn_map.get(i, 0),
            fn_map.get(i, 0),
        )
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if precision + recall > 0
            else 0
        )
        acc = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn > 0 else 0
        print(
            f"class: {i} precision: {precision:.2f} recall: {recall:.2f} f1: {f1:.2f} acc: {acc:.2f}"
        )


def adaptive_tracing_eval(
    model,
    dataset_id,
    datasets_ood,
    val_ood_to_test,
    dataset_id_test,
    datasets_ood_test,
    batch,
    n_syscall,
    n_category,
    device,
    log_folder,
    use_event_model,
    use_latency_model,
    ordinal_latency,
    continuous_latency,
    event_root_cause,
    duration_root_cause,
    use_unique_thresh,
    analyze_rootCause,
    test_random_cases,
    batch_error_vectors=True,
    load_from_saved_vecs=False,
):
    # Create directories if necessary
    if not os.path.exists(f"{log_folder}/evaluation/at"):
        os.makedirs(f"{log_folder}/evaluation/at")

    # Initialize the Cross-Entropy loss
    criterion = CrossEntropyLoss(ignore_index=0, reduction="none")
    criterion_latency = (
        MSELoss(reduction="none")
        if continuous_latency
        else (
            BCEWithLogitsLoss(reduction="none")
            if ordinal_latency
            else CrossEntropyLoss(ignore_index=0, reduction="none")
        )
    )

    # Send the model and the loss on the GPU
    torch.cuda.set_device(device)
    model.cuda(device)
    criterion.cuda(device)
    criterion_latency.cuda(device)

    if continuous_latency:
        min_max_vals = torch.load(f"{log_folder}/min_max_vals.pt")
        min_vals = min_max_vals["min_vals"].to(device)
        max_vals = min_max_vals["max_vals"].to(device)

    # Evaluate model
    model.eval()
    ppl_id_valid = []
    ppl_id_test = []
    slw_id_valid = []
    slw_id_test = []

    with torch.no_grad():
        # Create the dataloader for VALIDATION set
        dataloader_id = DataLoader(
            dataset_id[1],
            batch_size=batch,
            collate_fn=collate_fn,
            num_workers=8,
            pin_memory=True,
        )

        start_time = perf_counter()
        tmp_count = 0
        for data, pad_mask, _, _ in dataloader_id:
            tmp_count += 1
            # Send tensors to device
            X = [x.to(device) for x in data[:-2]]
            y = data[-1].to(device)
            y_latency = data[-2].to(device)
            pad_mask = pad_mask.to(device)

            # Get prediction
            out, out_latency = model(*X, pad_mask, chk=False)

            if use_event_model:
                loss = criterion(out.reshape(-1, n_syscall), y.reshape(-1))
                loss = loss.reshape(y.shape)
                loss = torch.sum(loss, 1) / torch.sum(~pad_mask, 1)
                ppl_id_valid.extend(loss.cpu().detach().tolist())

            if use_latency_model:
                # mask for durations with zero value (ignored)
                z_mask = y_latency != 0
                merged_mask = ~pad_mask & z_mask
                if ordinal_latency:
                    y_latency = transform_ordinal(y_latency, n_category)
                    loss_latency = criterion_latency(
                        out_latency.reshape(-1, n_category - 2),
                        y_latency.reshape(-1, n_category - 2),
                    )
                    loss_latency = torch.sum(loss_latency.reshape(y_latency.shape), -1)
                    loss_latency *= merged_mask
                elif continuous_latency:
                    y_latency = (y_latency - min_vals[X[0]]) / (
                        max_vals[X[0]] - min_vals[X[0]]
                    )
                    # Handling cases where min and max are the same
                    y_latency[min_vals[X[0]] == max_vals[X[0]]] = 0
                    loss_latency = criterion_latency(
                        out_latency.reshape(-1), y_latency.reshape(-1)
                    )
                    loss_latency = loss_latency.reshape(y_latency.shape)
                else:
                    loss_latency = criterion_latency(
                        out_latency.reshape(-1, n_category), y_latency.reshape(-1)
                    )
                    loss_latency = loss_latency.reshape(y_latency.shape)
                epsilon = 1e-8  # Small value to prevent division by zero
                loss_latency = torch.sum(loss_latency, 1) / (
                    torch.sum(merged_mask, 1) + epsilon
                )
                slw_id_valid.extend(loss_latency.cpu().detach().tolist())

        end_time = perf_counter()
        # print(
        #     f"Execution time per batch for loss calculation: {(end_time - start_time)*1000/tmp_count} ms"
        # )

        start_time = perf_counter()
        tmp_count = 0
        # Create the dataloader for TEST set
        dataloader_id_test = DataLoader(
            dataset_id_test[1],
            batch_size=batch,
            collate_fn=collate_fn,
            num_workers=8,
            pin_memory=True,
        )
        for data, pad_mask, _, _ in dataloader_id_test:
            tmp_count += 1
            # Send tensors to device
            X = [x.to(device) for x in data[:-2]]
            y = data[-1].to(device)
            y_latency = data[-2].to(device)
            pad_mask = pad_mask.to(device)

            # Get prediction
            out, out_latency = model(*X, pad_mask, chk=False)

            if use_event_model:
                loss = criterion(out.reshape(-1, n_syscall), y.reshape(-1))
                loss = loss.reshape(y.shape)
                loss = torch.sum(loss, 1) / torch.sum(~pad_mask, 1)
                # Concatenate the computed values for the batch
                ppl_id_test.extend(loss.cpu().detach().tolist())

            if use_latency_model:
                # mask for durations with zero value (ignored)
                z_mask = y_latency != 0
                merged_mask = ~pad_mask & z_mask
                if ordinal_latency:
                    y_latency = transform_ordinal(y_latency, n_category)
                    loss_latency = criterion_latency(
                        out_latency.reshape(-1, n_category - 2),
                        y_latency.reshape(-1, n_category - 2),
                    )
                    loss_latency = torch.sum(loss_latency.reshape(y_latency.shape), -1)
                    loss_latency *= merged_mask
                elif continuous_latency:
                    y_latency = (y_latency - min_vals[X[0]]) / (
                        max_vals[X[0]] - min_vals[X[0]]
                    )
                    # Handling cases where min and max are the same
                    y_latency[min_vals[X[0]] == max_vals[X[0]]] = 0
                    loss_latency = criterion_latency(
                        out_latency.reshape(-1), y_latency.reshape(-1)
                    )
                    loss_latency = loss_latency.reshape(y_latency.shape)
                else:
                    loss_latency = criterion_latency(
                        out_latency.reshape(-1, n_category), y_latency.reshape(-1)
                    )
                    loss_latency = loss_latency.reshape(y_latency.shape)
                epsilon = 1e-8  # Small value to prevent division by zero
                loss_latency = torch.sum(loss_latency, 1) / (
                    torch.sum(merged_mask, 1) + epsilon
                )
                slw_id_test.extend(loss_latency.cpu().detach().tolist())
        end_time = perf_counter()
        # print(
        #     f"Execution time per batch for loss calculation: {(end_time - start_time)*1000/tmp_count} ms"
        # )

        if analyze_rootCause:
            # creating a vector representation for each request/sample fed to model to
            # later use for root-cause analysis. vectors are made using mispredicted events
            folder_err_vecs = "error_vectors-" + str(n_category)
            if not load_from_saved_vecs:
                for name, dataset_ood in datasets_ood.items():
                    # Create the dataloader
                    dataloader_ood = DataLoader(
                        dataset_ood,
                        batch_size=batch,
                        collate_fn=collate_fn,
                        num_workers=8,
                        pin_memory=True,
                    )

                    (
                        all_err_vector_event,
                        all_err_vector_duration,
                        all_err_vector_avg,
                    ) = ([], [], [])
                    for data, pad_mask, _, _ in dataloader_ood:
                        # Send tensors to device
                        X = [x.to(device) for x in data[:-2]]
                        y = data[-1].to(device)
                        y_latency = data[-2].to(device)
                        pad_mask = pad_mask.to(device)

                        # Get prediction
                        out, out_latency = model(*X, pad_mask, chk=False)
                        # form the error vector for event model
                        if event_root_cause:
                            predictions_event = torch.argmax(out, dim=-1)
                            errors_event = predictions_event != y
                            error_vector_event = torch.zeros(
                                n_syscall, dtype=torch.long
                            )
                        if duration_root_cause:
                            z_mask = y_latency != 0
                            merged_mask = ~pad_mask & z_mask
                            if ordinal_latency:
                                predictions_duration = (
                                    torch.sigmoid(out_latency) > 0.5
                                ).sum(dim=-1) + 1
                            else:
                                predictions_duration = torch.argmax(out_latency, dim=-1)
                            errors_duration = predictions_duration != y_latency
                            errors_duration *= merged_mask

                            error_vector_duration = torch.zeros(
                                n_syscall, dtype=torch.long
                            )
                        if batch_error_vectors:
                            for i in range(n_syscall):
                                if event_root_cause:
                                    error_vector_event[i] = (
                                        errors_event & (y == i)
                                    ).sum()
                                if duration_root_cause:
                                    # X[0] is the input sequence of event types
                                    error_vector_duration[i] = (
                                        errors_duration & (X[0] == i)
                                    ).sum()
                            if event_root_cause:
                                all_err_vector_event.append(error_vector_event)
                            if duration_root_cause:
                                all_err_vector_duration.append(error_vector_duration)
                            if event_root_cause and duration_root_cause:
                                # Average of the two error vectors
                                average_error_vector = (
                                    error_vector_event + error_vector_duration
                                ) / 2
                                all_err_vector_avg.append(average_error_vector)
                        else:
                            # TODO: add the duration and mean error vector
                            for idx in range(len(errors_event)):
                                sequence_errors = errors_event[idx]
                                sequence_labels = y[idx]
                                # Only perform operations where sequence_errors is True
                                unique_errors, counts = sequence_labels[
                                    sequence_errors
                                ].unique(return_counts=True)
                                error_vector_event[
                                    unique_errors.cpu().long()
                                ] = counts.cpu()
                                all_err_vector_event.append(error_vector_event)

                    # Save the error vectors
                    error_vec_types = ["event", "duration", "average"]
                    vector_data_mapping = {
                        "event": all_err_vector_event,
                        "duration": all_err_vector_duration,
                        "average": all_err_vector_avg,
                    }
                    if not os.path.exists(folder_err_vecs):
                        os.makedirs(folder_err_vecs)
                    for err_type in error_vec_types:
                        filename = os.path.join(
                            folder_err_vecs, name + "_error_vecs_" + err_type + ".pkl"
                        )
                        with open(filename, "wb") as file:
                            pickle.dump(vector_data_mapping[err_type], file)
                        # print(f"saved error vectors for {name} with type {err_type}")

            # Compute the mean error vector or cluster the error vectors for each OOD validation set
            # List of error vector types
            if event_root_cause and duration_root_cause:
                error_vec_types = ["event", "duration", "average"]
            elif event_root_cause:
                error_vec_types = ["event"]
            elif duration_root_cause:
                error_vec_types = ["duration"]
            cluster_based, use_kmeans = True, False
            ood_mean_err_vec = {et: {} for et in error_vec_types}
            ood_centroid_vec = {et: {} for et in error_vec_types}
            for name, dataset_ood in datasets_ood.items():
                for err_type in error_vec_types:
                    # Load error vectors
                    filename = os.path.join(
                        folder_err_vecs, name + "_error_vecs_" + err_type + ".pkl"
                    )
                    with open(filename, "rb") as file:
                        all_error_vectors = pickle.load(file)
                    all_error_vectors = torch.stack(all_error_vectors)
                    # Cluster-based
                    if cluster_based:
                        if use_kmeans:
                            kmeans = KMeans(n_clusters=3)
                            kmeans.fit(all_error_vectors.numpy())
                            ood_centroid_vec[err_type][name] = kmeans.cluster_centers_
                        else:
                            import hdbscan

                            clusterer = hdbscan.HDBSCAN(min_cluster_size=2)
                            clusterer.fit(all_error_vectors.numpy())
                            ood_centroid_vec[err_type][name] = clusterer.exemplars_
                    # Average-based
                    else:
                        average_error_vector_ood = torch.mean(
                            all_error_vectors.float(), dim=0
                        )
                        ood_mean_err_vec[err_type][name] = average_error_vector_ood

        # find the best threshold for the OOD datasets
        d_vals_all, y_true_all = {}, {}
        median_mad_ppl, median_mad_slw = {}, {}
        for name, dataset_ood in datasets_ood.items():
            # For VALIDATION set:
            ppl_ood_valid = []
            slw_ood_valid = []

            # Create the dataloader
            dataloader_ood = DataLoader(
                dataset_ood,
                batch_size=batch,
                collate_fn=collate_fn,
                num_workers=8,
                pin_memory=True,
            )

            for data, pad_mask, _, _ in dataloader_ood:
                # Send tensors to device
                X = [x.to(device) for x in data[:-2]]
                y = data[-1].to(device)
                y_latency = data[-2].to(device)
                pad_mask = pad_mask.to(device)

                # Get prediction
                out, out_latency = model(*X, pad_mask, chk=False)

                if use_event_model:
                    loss = criterion(out.reshape(-1, n_syscall), y.reshape(-1))
                    loss = loss.reshape(y.shape)
                    loss = torch.sum(loss, 1) / torch.sum(~pad_mask, 1)
                    ppl_ood_valid.extend(loss.cpu().detach().tolist())

                if use_latency_model:
                    # mask for durations with zero value (ignored)
                    z_mask = y_latency != 0
                    merged_mask = ~pad_mask & z_mask
                    if ordinal_latency:
                        y_latency = transform_ordinal(y_latency, n_category)
                        loss_latency = criterion_latency(
                            out_latency.reshape(-1, n_category - 2),
                            y_latency.reshape(-1, n_category - 2),
                        )
                        loss_latency = torch.sum(
                            loss_latency.reshape(y_latency.shape), -1
                        )
                        loss_latency *= merged_mask
                    elif continuous_latency:
                        y_latency = (y_latency - min_vals[X[0]]) / (
                            max_vals[X[0]] - min_vals[X[0]]
                        )
                        # Handling cases where min and max are the same
                        y_latency[min_vals[X[0]] == max_vals[X[0]]] = 0
                        loss_latency = criterion_latency(
                            out_latency.reshape(-1), y_latency.reshape(-1)
                        )
                        loss_latency = loss_latency.reshape(y_latency.shape)
                    else:
                        loss_latency = criterion_latency(
                            out_latency.reshape(-1, n_category), y_latency.reshape(-1)
                        )
                        loss_latency = loss_latency.reshape(y_latency.shape)
                    epsilon = 1e-8  # Small value to prevent division by zero
                    loss_latency = torch.sum(loss_latency, 1) / (
                        torch.sum(merged_mask, 1) + epsilon
                    )
                    slw_ood_valid.extend(loss_latency.cpu().detach().tolist())

            median_ppl, median_slw = None, None
            mad_ppl, mad_slw = None, None
            if use_event_model and use_latency_model:
                m_len = min(len(ppl_id_valid), len(ppl_ood_valid))
                ppl = ppl_id_valid[:m_len] + ppl_ood_valid[:m_len]
                slw = slw_id_valid[:m_len] + slw_ood_valid[:m_len]
                # Normalize the losses using the MAD
                median_ppl = np.median(ppl)
                median_slw = np.median(slw)
                mad_ppl = np.median(np.abs(ppl - median_ppl))
                mad_slw = np.median(np.abs(slw - median_slw))
                ppl = (ppl - median_ppl) / mad_ppl
                slw = (slw - median_slw) / mad_slw
                d_vals = [p + s for p, s in zip(ppl, slw)]
                y_true = [0] * m_len + [1] * m_len
                median_mad_ppl.update({name: (median_ppl, mad_ppl)})
                median_mad_slw.update({name: (median_slw, mad_slw)})
            elif use_latency_model and not use_event_model:
                m_len = min(len(slw_id_valid), len(slw_ood_valid))
                d_vals = slw_id_valid[:m_len] + slw_ood_valid[:m_len]
                y_true = [0] * m_len + [1] * m_len
            else:
                m_len = min(len(ppl_id_valid), len(ppl_ood_valid))
                d_vals = ppl_id_valid[:m_len] + ppl_ood_valid[:m_len]
                y_true = [0] * m_len + [1] * m_len

            d_vals_all.update({name: d_vals})
            y_true_all.update({name: y_true})

        thresholds = {}
        if use_unique_thresh:
            combined_d_vals = list(itertools.chain(*d_vals_all.values()))
            combined_y_true = list(itertools.chain(*y_true_all.values()))

            fscore = []
            tmp_thresholds = np.arange(
                min(combined_d_vals),
                max(combined_d_vals),
                step=(max(combined_d_vals) - min(combined_d_vals)) / 200,
            )
            for t in tmp_thresholds:
                y_pred = [1 if p > t else 0 for p in combined_d_vals]
                fscore.append(sklearn.metrics.f1_score(combined_y_true, y_pred))

            id_best_threshold = np.argmax(fscore)
            best_threshold = tmp_thresholds[id_best_threshold]
            # set the same threshold for all OOD datasets
            thresholds = {name: best_threshold for name in datasets_ood.keys()}

            y_pred = [1 if p > best_threshold else 0 for p in combined_d_vals]
            print_classification_scores(
                combined_y_true, combined_d_vals, y_pred, "ALL VALID SETS"
            )
        else:
            for name in datasets_ood.keys():
                d_vals = d_vals_all.get(name)
                y_true = y_true_all.get(name)

                fscore = []
                tmp_thresholds = np.arange(
                    min(d_vals),
                    max(d_vals),
                    step=(max(d_vals) - min(d_vals)) / 200,
                )
                for t in tmp_thresholds:
                    y_pred = [1 if p > t else 0 for p in d_vals]
                    fscore.append(sklearn.metrics.f1_score(y_true, y_pred))

                id_best_threshold = np.argmax(fscore)
                best_threshold = tmp_thresholds[id_best_threshold]
                thresholds.update({name: best_threshold})

                y_pred = [1 if p > best_threshold else 0 for p in d_vals]
                print_classification_scores(y_true, d_vals, y_pred, name)

        # test the threshold on each of the test sets
        for name, dataset_ood in datasets_ood.items():
            # For TEST set:
            ppl_ood_test = []
            slw_ood_test = []
            ood_time_stamps = []

            # Get respective test data set
            name_ood_test = val_ood_to_test[name]
            dataset_ood_test = datasets_ood_test[name_ood_test]

            # Create the dataloader
            dataloader_ood = DataLoader(
                dataset_ood_test,
                batch_size=batch,
                collate_fn=collate_fn,
                num_workers=8,
                pin_memory=True,
            )

            root_cause_results = (
                {evt: [] for evt in error_vec_types} if analyze_rootCause else None
            )
            count_correct, count_wrong = 0, 0
            root_cause_latencies = []
            for data, pad_mask, _time_stamps, _ in dataloader_ood:
                # Send tensors to device
                X = [x.to(device) for x in data[:-2]]
                y = data[-1].to(device)
                y_latency = data[-2].to(device)
                pad_mask = pad_mask.to(device)

                # Get prediction
                out, out_latency = model(*X, pad_mask, chk=False)

                if use_event_model:
                    loss = criterion(out.reshape(-1, n_syscall), y.reshape(-1))
                    loss = loss.reshape(y.shape)
                    loss = torch.sum(loss, 1) / torch.sum(~pad_mask, 1)
                    ppl_ood_test.extend(loss.cpu().detach().tolist())

                if use_latency_model:
                    # mask for durations with zero value (ignored)
                    z_mask = y_latency != 0
                    merged_mask = ~pad_mask & z_mask
                    if ordinal_latency:
                        y_latency = transform_ordinal(y_latency, n_category)
                        loss_latency = criterion_latency(
                            out_latency.reshape(-1, n_category - 2),
                            y_latency.reshape(-1, n_category - 2),
                        )
                        loss_latency = torch.sum(
                            loss_latency.reshape(y_latency.shape), -1
                        )
                        loss_latency *= merged_mask
                    elif continuous_latency:
                        y_latency = (y_latency - min_vals[X[0]]) / (
                            max_vals[X[0]] - min_vals[X[0]]
                        )
                        # Handling cases where min and max are the same
                        y_latency[min_vals[X[0]] == max_vals[X[0]]] = 0
                        loss_latency = criterion_latency(
                            out_latency.reshape(-1), y_latency.reshape(-1)
                        )
                        loss_latency = loss_latency.reshape(y_latency.shape)
                    else:
                        loss_latency = criterion_latency(
                            out_latency.reshape(-1, n_category), y_latency.reshape(-1)
                        )
                        loss_latency = loss_latency.reshape(y_latency.shape)
                    epsilon = 1e-8  # Small value to prevent division by zero
                    loss_latency = torch.sum(loss_latency, 1) / (
                        torch.sum(merged_mask, 1) + epsilon
                    )
                    slw_ood_test.extend(loss_latency.cpu().detach().tolist())

                if test_random_cases:
                    if None not in _time_stamps:
                        # Collect the time stamps of end of each request
                        ood_time_stamps.extend(np.array(_time_stamps) / 1e6)
                    else:
                        ood_time_stamps.extend(np.full(batch, 0, dtype=int))

                # root-cause analysis
                if analyze_rootCause:
                    start_time = perf_counter()
                    if event_root_cause:
                        predictions_event = torch.argmax(out, dim=-1)
                        errors_event = predictions_event != y
                    if duration_root_cause:
                        if ordinal_latency:
                            predictions_duration = (
                                torch.sigmoid(out_latency) > 0.5
                            ).sum(dim=-1) + 1
                            y_latency = y_latency.sum(dim=-1) + 1
                        else:
                            predictions_duration = torch.argmax(out_latency, dim=-1)
                        errors_duration = predictions_duration != y_latency
                        errors_duration *= merged_mask

                    if batch_error_vectors:
                        # form the error vector for each batch
                        error_vector_event = torch.zeros(n_syscall, dtype=torch.long)
                        error_vector_duration = torch.zeros(n_syscall, dtype=torch.long)
                        for i in range(n_syscall):
                            if event_root_cause:
                                error_vector_event[i] = (errors_event & (y == i)).sum()
                            if duration_root_cause:
                                error_vector_duration[i] = (
                                    errors_duration & (X[0] == i)
                                ).sum()
                        if event_root_cause and duration_root_cause:
                            average_error_vector = (
                                error_vector_event + error_vector_duration
                            ) / 2
                    else:
                        # TODO: add the duration and mean error vector
                        # form the error vector for each individual sequence
                        for idx in range(len(errors_event)):
                            error_vector = torch.zeros(n_syscall, dtype=torch.long)
                            sequence_errors = errors_event[idx]
                            sequence_labels = y[idx]
                            # Only perform operations where sequence_errors is True
                            unique_errors, counts = sequence_labels[
                                sequence_errors
                            ].unique(return_counts=True)
                            error_vector[unique_errors.cpu().long()] = counts.cpu()

                            res = get_predictions(
                                error_vector,
                                ood_centroid_vec,
                                ood_mean_err_vec,
                                cluster_based,
                                use_kmeans,
                            )
                            if res == name:
                                count_correct += 1
                            else:
                                count_wrong += 1

                    for evt in error_vec_types:
                        if batch_error_vectors:
                            if evt == "average":
                                error_vector = average_error_vector
                            elif evt == "event":
                                error_vector = error_vector_event
                            else:
                                error_vector = error_vector_duration
                            res = get_predictions(
                                error_vector,
                                ood_centroid_vec[evt],
                                ood_mean_err_vec[evt],
                                cluster_based,
                                use_kmeans,
                            )
                            root_cause_results[evt].append(res == name)

                    end_time = perf_counter()
                    root_cause_latencies.append((end_time - start_time) * 1000)
            if analyze_rootCause:
                average_latency = (
                    sum(root_cause_latencies) / len(root_cause_latencies)
                    if root_cause_latencies
                    else 0
                )
                # print(
                #     f"\nExecution time per batch for root cause analysis: {average_latency} ms"
                # )
                print(
                    "\n" + name_ood_test,
                    "- root cause analysis results (separated from change detection):",
                )
                for evt in error_vec_types:
                    tmp_correct = sum(root_cause_results[evt])
                    tmp_len = len(root_cause_results[evt])
                    print(
                        "Error vector type: ",
                        evt,
                        "accuracy: ",
                        (tmp_correct / tmp_len) * 100,
                        "count: ",
                        tmp_len,
                    )

            start_time = perf_counter()
            if use_event_model and use_latency_model:
                m_len = min(len(ppl_id_test), len(ppl_ood_test))
                ppl_test = ppl_id_test[:m_len] + ppl_ood_test[:m_len]
                slw_test = slw_id_test[:m_len] + slw_ood_test[:m_len]
                # normalize
                # get median and mad from validation set
                median_ppl = median_mad_ppl.get(name)[0]
                median_slw = median_mad_slw.get(name)[0]
                mad_ppl = median_mad_ppl.get(name)[1]
                mad_slw = median_mad_slw.get(name)[1]

                ppl_test = (ppl_test - median_ppl) / mad_ppl
                slw_test = (slw_test - median_slw) / mad_slw
                d_vals_test = [p + s for p, s in zip(ppl_test, slw_test)]
                y_true = [0] * m_len + [1] * m_len
            elif use_latency_model and not use_event_model:
                m_len = min(len(slw_id_test), len(slw_ood_test))
                d_vals_test = slw_id_test[:m_len] + slw_ood_test[:m_len]
                y_true = [0] * m_len + [1] * m_len
            else:
                m_len = min(len(ppl_id_test), len(ppl_ood_test))
                d_vals_test = ppl_id_test[:m_len] + ppl_ood_test[:m_len]
                y_true = [0] * m_len + [1] * m_len

            y_pred = [1 if p > thresholds.get(name) else 0 for p in d_vals_test]
            end_time = perf_counter()
            # print(
            #     f"\nExecution time per batch for change detection: {(end_time - start_time)/(m_len/batch) * 1000} ms"
            # )
            print_classification_scores(y_true, d_vals_test, y_pred, name_ood_test)

            if use_event_model and use_latency_model:
                # normalize
                ppl_ood_test_norm = (ppl_ood_test[:m_len] - median_ppl) / mad_ppl
                ppl_id_test_norm = (ppl_id_test[:m_len] - median_ppl) / mad_ppl
                slw_ood_test_norm = (slw_ood_test[:m_len] - median_slw) / mad_slw
                slw_id_test_norm = (slw_id_test[:m_len] - median_slw) / mad_slw

                ood_test = [p + s for p, s in zip(ppl_ood_test_norm, slw_ood_test_norm)]
                id_test = [p + s for p, s in zip(ppl_id_test_norm, slw_id_test_norm)]
            elif use_latency_model and not use_event_model:
                ood_test = slw_ood_test[:m_len]
                id_test = slw_id_test[:m_len]
            else:
                ood_test = ppl_ood_test[:m_len]
                id_test = ppl_id_test[:m_len]

            # measure root cause classification considering the change detection accuracy
            if analyze_rootCause:
                y_pred = [True if p > thresholds.get(name) else False for p in ood_test]
                if batch_error_vectors:
                    for evt in error_vec_types:
                        root_cause_results[evt] = [
                            element
                            for element in root_cause_results[evt]
                            for _ in range(batch)
                        ]
                print(
                    name_ood_test,
                    "- Root cause analysis results (considering change detection):",
                )
                for evt in error_vec_types:
                    tmp_results = [
                        i and j for i, j in zip(y_pred, root_cause_results[evt])
                    ]
                    tmp_correct = sum(tmp_results)
                    tmp_len = len(tmp_results)
                    print(
                        "Error vector type: ",
                        evt,
                        "accuracy: ",
                        (tmp_correct / tmp_len) * 100,
                        "count: ",
                        tmp_len,
                    )

            # adaptive tracing on the test set
            if test_random_cases:
                ood_time_stamps, ood_test = zip(*sorted(zip(ood_time_stamps, ood_test)))

                print(
                    "\n"+name_ood_test, "- Adaptive Tracing:",
                )

                missed_anomalies, trace_reductions, delays = [], [], []
                for ratio in [i / 10 for i in range(11)]:  # 0.0, 0.1, 0.2, ..., 1.0
                    _missed_anomalies, _trace_reduction, _delay = [], [], []
                    for _ in range(30):
                        random_test_trace, y_true, t_stamps = create_test_scenario(
                            id_test, ood_test, ood_time_stamps, ratio
                        )

                        random_test_trace = random_test_trace[
                            : len(random_test_trace) // batch * batch
                        ]
                        y_true = y_true[: len(y_true) // batch * batch]

                        y_pred = [
                            1 if p > thresholds.get(name) else 0
                            for p in random_test_trace
                        ]

                        # decide to trace or not based on the predictions about requests
                        trace_decision, delay_history, tmp_dly1, tmp_dly2, tmp_dly3 = (
                            [],
                            [],
                            [],
                            [],
                            [],
                        )
                        window_size = batch
                        tracing = False
                        t_thresh = 0.8
                        last_traced_idx = 0
                        for i in range(0, len(y_pred) - window_size + 1, window_size):
                            window = y_pred[i : i + window_size]
                            do_trace = (
                                1
                                if (
                                    sum(elem == 1 for elem in window)
                                    > len(window) * t_thresh
                                )
                                else 0
                            )
                            # check if the tracer should have recorded this window
                            correct_trace = (
                                sum(e == 1 for e in y_true[i : i + window_size])
                                > window_size * t_thresh
                            )
                            # if the tracer should have recorded this window, and
                            # it just started recording, calculate the delay (delay
                            # related to the anomalies that model correctly predicted)
                            if do_trace == 1 and tracing == False and correct_trace:
                                if y_true[i] != 0:
                                    noise_start_idx = i
                                    while y_true[noise_start_idx - 1] != 0 and (
                                        noise_start_idx - 1 > last_traced_idx
                                    ):
                                        noise_start_idx -= 1
                                else:
                                    noise_start_idx = i
                                    while y_true[noise_start_idx] == 0:
                                        noise_start_idx += 1
                                noise_end_idx = i + window_size - 1
                                while y_true[noise_end_idx] != 1:
                                    noise_end_idx -= 1
                                delay_history.append(
                                    t_stamps[noise_end_idx] - t_stamps[noise_start_idx]
                                )
                            if do_trace == 1:
                                tracing = True
                                last_traced_idx = i + window_size - 1
                            else:
                                tracing = False
                            trace_decision.extend([do_trace] * window_size)

                        # assess the accuracy of the trace
                        tn, fp, fn, tp = confusion_matrix(
                            y_true, trace_decision, labels=[0, 1]
                        ).ravel()
                        _missed_anomalies.append(fn / (tp + fn) if tp + fn != 0 else 0)
                        _trace_reduction.append(1 - (tp + fp) / len(y_true))
                        if delay_history:
                            _delay.append(np.mean(delay_history))
                        else:
                            _delay.append(0.0)

                    mean_missed_anomalies = np.mean(_missed_anomalies)
                    mean_trace_reduction = np.mean(_trace_reduction)
                    mean_delay = np.mean(_delay)

                    print(
                        "ratio: {:.2f}, missed_anomalies (%): {:.2f}, trace_reduction (%): {:.2f}, delay (ms): {:.2f}".format(
                            ratio,
                            mean_missed_anomalies * 100,
                            mean_trace_reduction * 100,
                            mean_delay,
                        )
                    )

                    missed_anomalies.append(mean_missed_anomalies)
                    trace_reductions.append(mean_trace_reduction)
                    delays.append(mean_delay)
                print(
                    "Aggregated results: \nmissed_anomalies (%): {:.2f}, trace_reduction (%): {:.2f}, delay (ms): {:.2f}".format(
                        np.mean(missed_anomalies) * 100,
                        np.mean(trace_reductions) * 100,
                        np.mean(delays),
                    )
                )
                plot_random_test_traces(
                    [i / 10 for i in range(11)],
                    missed_anomalies,
                    trace_reductions,
                    delays,
                    name_ood_test,
                    log_folder,
                    use_event_model,
                    use_latency_model,
                )


def plot_random_test_traces(
    ratio,
    missed_anomalies,
    trace_reduction,
    delays,
    name,
    log_folder,
    use_event_model,
    use_latency_model,
):
    # Create directories if necessary
    if not os.path.exists(f"{log_folder}/evaluation/random-trace-test"):
        os.makedirs(f"{log_folder}/evaluation/random-trace-test")

    # convert noise to normal ration to noise percentage
    noise_percentages = (np.array(ratio) * 100 / 2).astype(int)

    # Plotting
    plt.figure(figsize=(10, 6))

    ax1 = plt.gca()
    ax2 = ax1.twinx()

    # Plot metrics on primary y-axis
    ax1.plot(
        noise_percentages,
        missed_anomalies,
        marker="o",
        color="blue",
        label="Miss Rate (%)",
    )
    ax1.plot(
        noise_percentages,
        trace_reduction,
        marker="o",
        color="green",
        label="Trace Reduction (%)",
    )
    # Plot delay on secondary y-axis
    ax2.plot(
        noise_percentages,
        delays,
        marker="o",
        color="red",
        linestyle="--",
        label="Delay (ms)",
    )

    ax1.set_xlabel("Trace Noise Level (%)")
    ax1.set_ylabel("Value (%)", color="blue")
    ax2.set_ylabel("Delay (ms)", color="red")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax1.grid(True)

    ax1.tick_params(axis="y", colors="blue")
    ax2.tick_params(axis="y", colors="red")

    plt.tight_layout()
    name = name.replace(" ", "_").lower()
    suffix = (
        "_event_duration"
        if use_event_model and use_latency_model
        else "_event"
        if use_event_model
        else "_duration"
    )
    plt.savefig(
        f"{log_folder}/evaluation/random-trace-test/{name}{suffix}.pdf",
        format="pdf",
        dpi=300,
    )

    plt.close()


def create_test_scenario(id_requests, ood_requests, ood_t_stamps, noise_normal_ratio):
    mixed_calls = []
    tags = []
    time_stamps = []

    ood_size = int(noise_normal_ratio * len(id_requests))
    rnd_start_idx = random.randint(0, len(ood_requests) - ood_size)
    ood_requests = ood_requests[rnd_start_idx : (rnd_start_idx + ood_size)]

    id_len, ood_len = len(id_requests), len(ood_requests)

    while id_requests or ood_requests:
        # Add a random number of normal events to mixed set
        num_normal = random.randint(1, max(1, int(0.1 * id_len)))
        mixed_calls.extend(id_requests[:num_normal])
        tags.extend([0] * len(id_requests[:num_normal]))
        time_stamps.extend([0] * len(id_requests[:num_normal]))
        id_requests = id_requests[num_normal:]

        # Decide if to add a noisy burst
        if random.random() < 0.5 and ood_len > 0:
            ood_burst_size = random.randint(1, max(1, int(0.2 * ood_len)))
            mixed_calls.extend(ood_requests[:ood_burst_size])
            tags.extend([1] * len(ood_requests[:ood_burst_size]))
            time_stamps.extend(ood_t_stamps[: len(ood_requests[:ood_burst_size])])
            ood_requests = ood_requests[ood_burst_size:]
            ood_t_stamps = ood_t_stamps[ood_burst_size:]

    return mixed_calls, tags, time_stamps


def print_classification_scores(y_true, d_vals, y_pred, name):
    precision = sklearn.metrics.precision_score(y_true, y_pred)
    recall = sklearn.metrics.recall_score(y_true, y_pred)
    accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
    fscore = sklearn.metrics.f1_score(y_true, y_pred)

    # Log
    auroc = sklearn.metrics.roc_auc_score(y_true, d_vals)
    print(f"{name}:")
    print(f"{'    AUROC':30}: {auroc:68.2%}")
    print(f"{'    Recall':30}: {recall:68.2%}")
    print(f"{'    Precision':30}: {precision:68.2%}")
    print(f"{'    F-score':30}: {np.max(fscore):68.2%}")
    print(f"{'    Accuracy':30}: {accuracy:68.2%}")


def get_predictions(
    error_vector, ood_centroid_vec, ood_mean_err_vec, cluster_based, use_kmeans
):
    if cluster_based:
        if use_kmeans:
            res = predict_class_centroid(error_vector, ood_centroid_vec)
        else:
            res = predict_class_hdbscan(error_vector, ood_centroid_vec)
    else:
        res = predict_class_mean(error_vector, ood_mean_err_vec)
    return res


def predict_class_hdbscan(new_vector, class_centroids):
    min_distance = float("inf")
    best_class = None

    for class_name, exemplars in class_centroids.items():
        for exemplar in exemplars:
            distance = np.min(
                cdist(new_vector.reshape(1, -1), exemplar, metric="euclidean")
            )
            if distance < min_distance:
                min_distance = distance
                best_class = class_name
    return best_class


def predict_class_mean(new_vector, class_vectors):
    # Initialize max similarity and predicted class
    max_similarity = -1
    predicted_class = None

    # Go through each class and its corresponding vector
    for class_name, class_vector in class_vectors.items():
        # Calculate cosine similarity between new vector and current class vector
        similarity = cosine_similarity(
            new_vector.unsqueeze(0), class_vector.unsqueeze(0)
        )

        # If this class's similarity is higher than the current max, update max and prediction
        if similarity > max_similarity:
            max_similarity = similarity
            predicted_class = class_name

    return predicted_class


def predict_class_centroid(new_vector, class_centroids):
    # Convert the new vector to a numpy array
    new_vector_array = new_vector.numpy()

    min_distance = float("inf")
    predicted_class = None

    for class_name, centroids in class_centroids.items():
        for centroid in centroids:
            distance = np.linalg.norm(new_vector_array - centroid)

            if distance < min_distance:
                min_distance = distance
                predicted_class = class_name

    return predicted_class


def plot_grad_norm(grad_norm, clip, log_folder):
    # Create figure
    fig = plt.figure(figsize=(10, 6), tight_layout=True)
    ax = fig.add_subplot(111)
    # Set colors
    dark_gray = "#808080"
    # Hide the right and top spines
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # Change axes and tick color
    ax.spines["bottom"].set_color(dark_gray)
    ax.tick_params(axis="x", colors=dark_gray)
    ax.spines["left"].set_color(dark_gray)
    ax.tick_params(axis="y", colors=dark_gray)
    ax.xaxis.label.set_color(dark_gray)
    ax.yaxis.label.set_color(dark_gray)
    # Plot
    plt.plot(range(1, len(grad_norm) + 1), grad_norm, lw=1)
    if clip is not None:
        plt.axhline(clip, lw=1, c=dark_gray)
    # Labels
    plt.xlabel("Update")
    plt.ylabel("Gradient Norm")
    # Make title the length of the graph
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="11%", pad=0)
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    for x in cax.spines:
        cax.spines[x].set_visible(False)
    cax.spines["top"].set_visible(False)
    cax.set_facecolor(dark_gray)
    at = AnchoredText(
        "Gradient L2-Norm",
        loc=6,
        pad=0,
        prop=dict(backgroundcolor=dark_gray, size=18, color="w"),
    )
    at.patch.set_edgecolor("none")
    cax.add_artist(at)
    # Save figure
    plt.savefig(f"{log_folder}/training/grad_norm.png", dpi=300)
    plt.close("all")


###############################################################################
# Visualization
###############################################################################


def plot_hist(x, mapping, dir, name, log_folder):
    """Plot the histogram of system call or process names

    Args:
        x (list): Dataset of system call or process names.
        mapping (dict): Mapping from int to string.
        dir (str): Name of the dataset.
        name (str): Name (syscall/process).
        it (int): Iteration number (log folder).
    """
    # Count the number of occurence of each word
    count = [0 for _ in mapping]
    for _x in x:
        for w in _x:
            count[int(w)] += 1
    # Convert to probability
    count = [c / sum(count) for c in count]
    # Sort and keep the 10 most probable
    count, mapping = map(list, zip(*sorted(zip(count, mapping))))
    count = count[-9:]
    mapping = mapping[-9:]
    # Add 'other'
    count.insert(0, 1 - sum(count))
    mapping.insert(0, "other")
    # Create figure
    fig = plt.figure(figsize=(10, 6), tight_layout=True)
    ax = fig.add_subplot(111)
    # Set colors
    dark_gray = "#808080"
    light_gray = "#D3D3D3"
    # Plot
    bins = [x - 0.5 for x in range(len(mapping) + 1)]
    n, bins, patches = plt.hist(
        mapping, bins=bins, weights=count, rwidth=0.8, orientation="horizontal"
    )
    # Hide the bottom, right and top spines and ticks
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.tick_params(axis="y", which="both", left=False, right=False, labelleft=True)
    # Change color of other
    patches[0].set_fc(light_gray)
    # For each bar: Place a label
    for i, (c, p) in enumerate(zip(count, patches)):
        x_value = p.get_width()
        y_value = p.get_y() + p.get_height() / 2
        if x_value > 0.01:
            plt.annotate(
                "{:.0%}".format(c),
                (x_value, y_value),
                color="w" if i != 0 else "k",
                xytext=(-2, 0),
                textcoords="offset points",
                va="center",
                ha="right",
            )
    # Change colors and labels of Y axis
    ax.spines["left"].set_color(dark_gray)
    # Add the name to the y-axis
    ax.tick_params(axis="y", colors=dark_gray, labelsize=14)
    ax.set_yticks(range(len(mapping)))
    ax.set_yticklabels(mapping)
    ax.tick_params(axis="x", colors="w")
    # Make title the length of the graph
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="11%", pad=0)
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    for x in cax.spines:
        cax.spines[x].set_visible(False)
    cax.spines["top"].set_visible(False)
    cax.set_facecolor(dark_gray)
    at = AnchoredText(
        f"{name} Names in the {dir}",
        loc=6,
        pad=0,
        prop=dict(backgroundcolor=dark_gray, size=18, color="w"),
    )
    at.patch.set_edgecolor("none")
    cax.add_artist(at)
    # Save figurex
    name = name.replace(" ", "_").lower()
    plt.savefig(f"{log_folder}/datasets/{dir}/hist_{name}.png", dpi=300)
    plt.close("all")


def plot_loss(train, val, eval, log_folder):
    """Plot the loss as a function of the model updates.

    Args:
        train (list): Losses on the training set.
        val (list): Losses on the validation set.
        eval (int): Number of updates between two evaluations.
        it (int): Iteration number (log folder).
    """
    # Create figure
    fig = plt.figure(figsize=(10, 6), tight_layout=True)
    ax = fig.add_subplot(111)
    # Set colors
    dark_gray = "#808080"
    # Hide the right and top spines
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # Change axes and tick color
    ax.spines["bottom"].set_color(dark_gray)
    ax.tick_params(axis="x", colors=dark_gray)
    ax.spines["left"].set_color(dark_gray)
    ax.tick_params(axis="y", colors=dark_gray)
    ax.xaxis.label.set_color(dark_gray)
    ax.yaxis.label.set_color(dark_gray)
    # Plot
    x = range(eval, (len(train) + 1) * eval, eval)
    ax.plot(x, train, color="C0")
    ax.annotate(
        "Train {:6.3f}".format(train[-1]),
        xy=(x[-1], train[-1]),
        xytext=(5, -5 if train[-1] < val[-1] else 5),
        size=12,
        textcoords="offset points",
        va="center",
        color="C0",
    )
    ax.plot(x, val, color="C1")
    ax.annotate(
        "Valid {:6.3f}".format(val[-1]),
        xy=(x[-1], val[-1]),
        xytext=(5, 5 if train[-1] < val[-1] else -5),
        size=12,
        textcoords="offset points",
        va="center",
        color="C1",
    )
    # Increase left margin
    lim = ax.get_xlim()
    right = lim[1] + (lim[1] - lim[0]) * 0.1
    ax.set_xlim(lim[0], right)
    # Labels
    plt.xlabel("Updates")
    plt.ylabel("Cross Entropy")
    # Make title the length of the graph
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="11%", pad=0)
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    for x in cax.spines:
        cax.spines[x].set_visible(False)
    cax.spines["top"].set_visible(False)
    cax.set_facecolor(dark_gray)
    at = AnchoredText(
        "Cross-entropy During Training",
        loc=6,
        pad=0,
        prop=dict(backgroundcolor=dark_gray, size=18, color="w"),
    )
    at.patch.set_edgecolor("none")
    cax.add_artist(at)
    # Save figure
    plt.savefig(f"{log_folder}/training/loss.png", dpi=300)
    plt.close("all")


def plot_accuracy(train, val, eval, log_folder):
    """Plot the accuracy as a function of the model updates.

    Args:
        train (list): Accuracies on the training set.
        val (list): Accuracies on the validation set.
        eval (int): Number of updates between two evaluations.
        it (int): Iteration number (log folder).
    """
    # Create figure
    fig = plt.figure(figsize=(10, 6), tight_layout=True)
    ax = fig.add_subplot(111)
    # Set colors
    dark_gray = "#808080"
    # Hide the right and top spines
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # Change axes and tick color
    ax.spines["bottom"].set_color(dark_gray)
    ax.tick_params(axis="x", colors=dark_gray)
    ax.spines["left"].set_color(dark_gray)
    ax.tick_params(axis="y", colors=dark_gray)
    ax.xaxis.label.set_color(dark_gray)
    ax.yaxis.label.set_color(dark_gray)
    # Plot
    x = range(eval, (len(train) + 1) * eval, eval)
    ax.plot(x, train, color="C0")
    ax.annotate(
        "Train {:6.1%}".format(train[-1]),
        xy=(x[-1], train[-1]),
        xytext=(5, -5 if train[-1] < val[-1] else 5),
        size=12,
        textcoords="offset points",
        va="center",
        color="C0",
    )
    ax.plot(x, val, color="C1")
    ax.annotate(
        "Valid {:6.1%}".format(val[-1]),
        xy=(x[-1], val[-1]),
        xytext=(5, 5 if train[-1] < val[-1] else -5),
        size=12,
        textcoords="offset points",
        va="center",
        color="C1",
    )
    # Increase left margin
    lim = ax.get_xlim()
    right = lim[1] + (lim[1] - lim[0]) * 0.1
    ax.set_xlim(lim[0], right)
    # Labels
    plt.xlabel("Updates")
    plt.ylabel("Accuracy")
    # Make title the length of the graph
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="11%", pad=0)
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    for x in cax.spines:
        cax.spines[x].set_visible(False)
    cax.spines["top"].set_visible(False)
    cax.set_facecolor(dark_gray)
    at = AnchoredText(
        "Accuracy During Training",
        loc=6,
        pad=0,
        prop=dict(backgroundcolor=dark_gray, size=18, color="w"),
    )
    at.patch.set_edgecolor("none")
    cax.add_artist(at)
    # Save figure
    plt.savefig(f"{log_folder}/training/accuracy.png", dpi=300)
    plt.close("all")


def plot_duration(duration, name, log_folder):
    """Plot the distribution of requests' duration.

    Args:
        duration (list): Requests' duration.
        name (str): Name of the dataset.
        it (int): Iteration number (log folder).
    """
    # Create figure
    fig = plt.figure(figsize=(10, 6), tight_layout=True)
    ax = fig.add_subplot(111)
    # Set colors
    dark_gray = "#808080"
    # Hide the right and top spines
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # Change axes and tick color
    ax.spines["bottom"].set_color(dark_gray)
    ax.tick_params(axis="x", colors=dark_gray)
    ax.spines["left"].set_color(dark_gray)
    ax.tick_params(axis="y", colors=dark_gray)
    ax.xaxis.label.set_color(dark_gray)
    ax.yaxis.label.set_color(dark_gray)
    # Plot the histogram
    plt.hist(duration, bins=50, rwidth=0.8, range=(0, 10))
    # Remove frame
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # Axis labels
    plt.xlabel("Request duration (ms)")
    plt.ylabel("Count")
    # Make title the length of the graph
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="11%", pad=0)
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    for x in cax.spines:
        cax.spines[x].set_visible(False)
    cax.spines["top"].set_visible(False)
    cax.set_facecolor(dark_gray)
    at = AnchoredText(
        f"Request Duration in the {name} Set",
        loc=6,
        pad=0,
        prop=dict(backgroundcolor=dark_gray, size=18, color="w"),
    )
    at.patch.set_edgecolor("none")
    cax.add_artist(at)
    # Save and close the figure
    plt.savefig(f"{log_folder}/datasets/{name}/hist_duration.png", dpi=300)
    plt.close("all")


def plot_length(length, name, log_folder):
    """Plot the distribution of requests' length.

    Args:
        length (list): Requests' length.
        name (str): Name of the dataset.
        it (int): Iteration number (log folder).
    """
    # Create figure
    fig = plt.figure(figsize=(10, 6), tight_layout=True)
    ax = fig.add_subplot(111)
    # Set colors
    dark_gray = "#808080"
    # Hide the right and top spines
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # Change axes and tick color
    ax.spines["bottom"].set_color(dark_gray)
    ax.tick_params(axis="x", colors=dark_gray)
    ax.spines["left"].set_color(dark_gray)
    ax.tick_params(axis="y", colors=dark_gray)
    ax.xaxis.label.set_color(dark_gray)
    ax.yaxis.label.set_color(dark_gray)
    # Plot the histogram
    plt.hist(length, bins=50, rwidth=0.8)
    # Remove frame
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # Axis labels
    plt.xlabel("Number of events")
    plt.ylabel("Count")
    # Make title the length of the graph
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="11%", pad=0)
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    for x in cax.spines:
        cax.spines[x].set_visible(False)
    cax.spines["top"].set_visible(False)
    cax.set_facecolor(dark_gray)
    at = AnchoredText(
        f"Request Length in the {name} Set",
        loc=6,
        pad=0,
        prop=dict(backgroundcolor=dark_gray, size=18, color="w"),
    )
    at.patch.set_edgecolor("none")
    cax.add_artist(at)
    # Save and close the figure
    plt.savefig(f"{log_folder}/datasets/{name}/hist_length.png", dpi=300)
    plt.close("all")
