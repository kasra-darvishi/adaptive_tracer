import torch.nn as nn
import torch

from . import Embedding


class LSTM(nn.Module):
    def __init__(
        self,
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
        train_event,
        train_latency,
        ordinal_latency,
        continuous_latency=False,
    ):
        super(LSTM, self).__init__()

        self.train_event = train_event
        self.train_latency = train_latency

        # Dropout
        dropout = 0 if dropout is None else dropout

        # Compute the embedding size
        self.d_model = sum(
            (
                dim_sys,
                dim_entry,
                dim_ret,
                dim_proc,
                dim_pid,
                dim_tid,
                dim_order,
                dim_time,
            )
        )

        # Embedding
        self.embedding = Embedding(
            n_syscall,
            n_category,
            n_process,
            dim_sys,
            dim_entry,
            dim_ret,
            dim_proc,
            dim_pid,
            dim_tid,
            dim_order,
            dim_time,
        )

        self.emb_dropout = nn.Dropout(dropout)

        # LSTM
        self.hidden_dim = n_hidden
        self.layers = n_layer
        self.lstm = nn.LSTM(
            input_size=self.d_model,
            hidden_size=self.hidden_dim,
            num_layers=self.layers,
            batch_first=True,
            dropout=dropout,
        )

        # Classifier
        self.classifier = nn.Linear(n_hidden, n_syscall) if train_event else None
        latency_linear_dim = 1 if continuous_latency else (n_category - 2 if ordinal_latency else n_category)
        self.classifier_latency = (nn.Linear(n_hidden, latency_linear_dim) if train_latency else None)

        self.init_weights()

    def init_weights(self):
        """Initialize the classifier weights using the uniform
        distribution proposed by Xavier & Bengio."""
        if self.train_event:
            nn.init.xavier_uniform_(self.classifier.weight)
            self.classifier.bias.data.zero_()
        if self.train_latency:
            nn.init.xavier_uniform_(self.classifier_latency.weight)
            self.classifier_latency.bias.data.zero_()

    def forward(self, call, entry, time, proc, pid, tid, ret, *args, **kwargs):
        src = self.embedding(call, entry, ret, time, proc, pid, tid)
        src = self.emb_dropout(src)
        h_t, _ = self.lstm(src)
        return self.classifier(h_t) if self.train_event else torch.empty(
            0
        ), self.classifier_latency(h_t) if self.train_latency else torch.empty(0)
