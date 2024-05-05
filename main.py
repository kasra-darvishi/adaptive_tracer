#!/usr/bin/env python
"""Main file.
"""
import os
import sys
import random
import numpy as np

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from dataset import Dictionary
from dataset import IterableDataset

from functions import (
    generate_dataset_request_based,
    adaptive_tracing_eval,
    nltk_ngram,
    ngram_eval,
    ood_detection_ngram,
    get_arguments,
    train,
    evaluate,
    dataset_stat,
)

from models import LSTM
from models import Transformer
import pickle

if __name__ == "__main__":

    ###########################################################################
    # Miscellaneous
    ###########################################################################

    # Get hyperparameters
    args = get_arguments()

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Get iteration number if necessary and create the corresponding folder
    if args.log_folder is None:
        iter = [int(f) for f in next(os.walk("logs"))[1]]
        args.log_folder = os.path.join("logs", max(iter) + 1 if iter else 0)
    if not os.path.exists(args.log_folder):
        os.makedirs(args.log_folder)

    # Redirect the output into a file (work with multiprocess)
    sys.stdout = open(f"{args.log_folder}/log.txt", "a", buffering=4096)

    # Log the arguments
    print(f"{'=' * 100}\n{'Arguments':^100s}\n{'=' * 100}")
    for arg in vars(args):
        if arg == "valid_ood_folders" or arg == "test_ood_folders":
            print(f"{arg:30s}: {str(getattr(args, arg)).split(',')[0]:>68}")
            for v in str(getattr(args, arg)).split(",")[1:]:
                print(f"{v:>100}")
        else:
            print(f"{arg:30s}: {str(getattr(args, arg)):>68}")

    ###########################################################################
    # Load data
    ###########################################################################

    # Paths to the traces
    paths = dict()
    # Train
    train_name = args.train_folder.split(":")[0]
    paths[train_name] = os.path.join(args.data_path, args.train_folder.split(":")[1])

    # Valid ID (In Distribution)
    valid_id_name = args.valid_id_folder.split(":")[0]
    paths[valid_id_name] = os.path.join(
        args.data_path, args.valid_id_folder.split(":")[1]
    )
    # Valid OOD (Out of Distribution)
    valid_ood_names = []
    for ood_folder in args.valid_ood_folders.split(","):
        valid_ood_names.append(ood_folder.split(":")[0])
        paths[ood_folder.split(":")[0]] = os.path.join(
            args.data_path, ood_folder.split(":")[1]
        )

    # Test ID (In Distribution)
    test_id_name = args.test_id_folder.split(":")[0]
    paths[test_id_name] = os.path.join(
        args.data_path, args.test_id_folder.split(":")[1]
    )
    # Test OOD (Out of Distribution)
    test_ood_names = []
    for test_folder in args.test_ood_folders.split(","):
        test_ood_names.append(test_folder.split(":")[0])
        paths[test_folder.split(":")[0]] = os.path.join(
            args.data_path, test_folder.split(":")[1]
        )

    # Linking Valid OODs data set to their respective test set
    val_ood_to_test = {
        valid: test for valid, test in zip(valid_ood_names, test_ood_names)
    }

    if args.generate_dataset:
        # Generate the train dataset and the dictionaries
        dict_sys, dict_proc = Dictionary(), Dictionary()
        event_delay_spans = generate_dataset_request_based(
            paths[train_name], dict_sys, dict_proc, args.n_categories, train=True
        )
        dict_sys.save(os.path.join(args.data_path, "dict_sys.pkl"))
        dict_proc.save(os.path.join(args.data_path, "dict_proc.pkl"))
        with open(os.path.join(args.data_path, "dict_delay_spans.pkl"), "wb") as f:
            pickle.dump(event_delay_spans, f)
        # with open(os.path.join(args.data_path, "dict_delay_spans.pkl"), "rb") as f:
        #     event_delay_spans = pickle.load(f)

        # Generate the other datasets without updating the dictionaries
        for name, path in paths.items():
            if name != train_name:
                generate_dataset_request_based(
                    path,
                    dict_sys,
                    dict_proc,
                    args.n_categories,
                    event_delay_spans=event_delay_spans,
                )

    else:
        dict_sys = Dictionary(os.path.join(args.data_path, "dict_sys.pkl"))
        dict_proc = Dictionary(os.path.join(args.data_path, "dict_proc.pkl"))

    # Update the paths to be the generated datasets
    for k, v in paths.items():
        paths[k] = os.path.join(v, "data.txt")

    cat_idx = (
        0
        if args.n_categories == 4
        else 1
        if args.n_categories == 6
        else 2
        if args.n_categories == 8
        else 3
    )

    # Create the datasets
    datasets = {
        k: IterableDataset(
            v,
            args.max_sample,
            args.max_token,
            cat_idx,
            args.multi_category,
            args.continuous_latency,
        )
        for k, v in paths.items()
    }

    ###########################################################################
    # Data analysis
    ###########################################################################

    n_syscall = len(dict_sys)
    n_process = len(dict_proc)

    print(f"{'=' * 100}\n{'Vocabulary':^100s}\n{'=' * 100}")
    print(f"{'Vocabulary size':30}: {n_syscall:68,}")
    print(f"{'Number of processes':30}: {n_process:68,}")

    if args.dataset_stat:
        for name, path in zip(datasets.keys(), paths.values()):
            dataset_stat(
                path,
                dict_sys,
                dict_proc,
                name,
                args.log_folder,
            )

    ###########################################################################
    # Build and train the model
    ###########################################################################

    # Ngram
    if args.model == "ngram":
        # Build the Ngram
        print(f"{'=' * 100}\n{'Model':^100s}\n{'=' * 100}")
        counter = nltk_ngram(paths[train_name], args.order, args.max_sample)

        # Evaluate on the datasets
        print(f"{'=' * 100}\n{'Evaluation':^100s}\n{'=' * 100}")
        for name, path in paths.items():
            ngram_eval(path, counter, args.order, name, args.max_sample)

        if args.analysis:
            print(f"{'=' * 100}\n{'OOD Detection':^100s}\n{'=' * 100}")

            # Separate dataset in-distribution (known behaviour) from
            # out-of-distribution (novel behaviour)
            path_valid_id = (valid_id_name, paths[valid_id_name])
            path_test_id = (test_id_name, paths[test_id_name])
            paths_valid_ood = {k: v for k, v in paths.items() if k in valid_ood_names}
            paths_test_ood = {k: v for k, v in paths.items() if k in test_ood_names}

            ood_detection_ngram(
                counter,
                args.order,
                1 / n_syscall,
                path_valid_id,
                paths_valid_ood,
                val_ood_to_test,
                path_test_id,
                paths_test_ood,
                args.log_folder,
                args.max_sample,
            )

        # Exit
        sys.exit(0)

    args.gpu = list(map(int, args.gpu.split(",")))

    # If the model is trained
    if args.load_model is None:

        # Prerequisites for Distributed Data Parallel
        # See https://yangkky.github.io/2019/07/08/
        # distributed-pytorch-tutorial.html
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "8888"

        # Create directories if necessary
        if not os.path.exists(f"{args.log_folder}/training"):
            os.makedirs(f"{args.log_folder}/training")

        try:
            # Spawn processes (worker)
            mp.spawn(
                train,
                nprocs=len(args.gpu),
                args=(
                    args.model,
                    n_syscall,
                    args.n_categories,
                    n_process,
                    args.n_head,
                    args.n_hidden,
                    args.n_layer,
                    args.dropout,
                    args.dim_sys,
                    args.dim_entry,
                    args.dim_ret,
                    args.dim_proc,
                    args.dim_pid,
                    args.dim_tid,
                    args.dim_order,
                    args.dim_time,
                    args.dim_f_mean,
                    args.activation,
                    args.tfixup,
                    datasets[train_name],
                    datasets[valid_id_name],
                    args.n_update,
                    args.reduce_lr_patience,
                    args.early_stopping_patience,
                    args.warmup_steps,
                    args.lr,
                    args.ls,
                    args.clip,
                    args.eval,
                    args.batch,
                    args.gpu,
                    args.chk,
                    args.amp,
                    args.log_folder,
                    args.train_event_model,
                    args.train_latency_model,
                    args.ordinal_latency,
                    args.continuous_latency,
                ),
            )
        except Exception as e:
            # Print the error message
            print(e)

    # Setting which GPU must be used as principal
    device = args.gpu[0]  # always the first in the list

    # Create a new model on a single GPU
    if args.model == "lstm":
        model = LSTM(
            n_syscall,
            args.n_categories,
            n_process,
            args.n_hidden,
            args.n_layer,
            args.dropout,
            args.dim_sys,
            args.dim_entry,
            args.dim_ret,
            args.dim_proc,
            args.dim_pid,
            args.dim_tid,
            args.dim_order,
            args.dim_time,
            args.dim_f_mean,
            args.train_event_model,
            args.train_latency_model,
            args.ordinal_latency,
            args.continuous_latency,
        ).to(args.gpu[0])

    if args.model == "transformer":
        model = Transformer(
            n_syscall,
            args.n_categories,
            n_process,
            args.n_head,
            args.n_hidden,
            args.n_layer,
            args.dropout,
            args.dim_sys,
            args.dim_entry,
            args.dim_ret,
            args.dim_proc,
            args.dim_pid,
            args.dim_tid,
            args.dim_order,
            args.dim_time,
            args.dim_f_mean,
            args.activation,
            args.tfixup,
            args.train_event_model,
            args.train_latency_model,
            args.ordinal_latency,
        ).to(args.gpu[0])

    # Load the best model's parameters or the one specified in argument
    modelfile = os.path.join(
        args.log_folder if args.load_model is None else args.load_model,
        "model",
    )
    with open(modelfile, "rb") as f:
        # Load the model on the GPU
        model.load_state_dict(torch.load(f))
        print("Model loaded")

    # Evaluate the model
    if args.eval_token_pred:
        print(f"{'=' * 100}\n{'Evaluation':^100s}\n{'=' * 100}")

        criterion = nn.CrossEntropyLoss(ignore_index=0)
        criterion_latency = (
            nn.MSELoss()
            if args.continuous_latency
            else (
                nn.BCEWithLogitsLoss()
                if args.ordinal_latency
                else nn.CrossEntropyLoss(ignore_index=0)
            )
        )

        for name, dataset in datasets.items():
            if "Valid" in name:
                continue
            loss, acc, loss_latency, acc_latency, mae_latency, _ = evaluate(
                model,
                dataset,
                args.batch,
                criterion,
                criterion_latency,
                n_syscall,
                args.n_categories,
                args.gpu[0],
                args.train_event_model,
                args.train_latency_model,
                args.ordinal_latency,
                args.continuous_latency,
            )
            print(
                f"{name:30}: {f'loss {loss:5.3f} acc {acc:5.1%} loss_latency {loss_latency:5.3f} acc_latency {acc_latency:5.1%} mae_latency {mae_latency:5.3f}':>68}"
            )

    ###########################################################################
    # Model analysis
    ###########################################################################

    if args.analysis:
        # Separate dataset in-distribution (known behaviour) from
        # out-of-distribution (novel behaviour)
        dataset_valid_id = (valid_id_name, datasets[valid_id_name])
        datasets_valid_ood = {k: v for k, v in datasets.items() if k in valid_ood_names}

        dataset_test_id = (test_id_name, datasets[test_id_name])
        datasets_test_ood = {k: v for k, v in datasets.items() if k in test_ood_names}

        test_conditions = None
        if args.train_event_model and args.train_latency_model:
            event_root_cause, duration_root_cause = True, False
        elif args.train_event_model:
            event_root_cause, duration_root_cause = True, False
        elif args.train_latency_model:
            event_root_cause, duration_root_cause = False, True
        # Detect out-of-distribution based on the cross-entropy loss
        print(
            f"{'=' * 100}\n{'OOD Detection':^100s}\n{'event_model='+str(args.train_event_model)+' & duration_model='+str(args.train_latency_model):^100s}\n{'=' * 100}"
        )
        adaptive_tracing_eval(
            model,
            dataset_valid_id,
            datasets_valid_ood,
            val_ood_to_test,
            dataset_test_id,
            datasets_test_ood,
            args.batch,
            n_syscall,
            args.n_categories,
            args.gpu[0],
            args.log_folder,
            args.train_event_model,
            args.train_latency_model,
            args.ordinal_latency,
            args.continuous_latency,
            event_root_cause,
            duration_root_cause,
            args.unique_thresh,
            args.analyze_rootCause,
            args.test_random_cases,
        )

    # Close the log file with a delimiter
    print("=" * 100)
