
import copy
import torch
import torch.nn as nn
import numpy as np
import itertools

from tabulate import tabulate
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

from .RETAIN import RETAIN

# from .anno_transformer.Modules import make_model

from utils.tensor_utils import fit_input_to_output
import torch.backends.cudnn as cudnn
cudnn.enabled = False

import sys
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

tabulate.PRESERVE_WHITESPACE = True
torch.manual_seed(5)


def masked_bce_loss(pred, trg, trg_len):
    """
    :param pred: prediction tensor size of (n_batch x max_seq_len x target_size)
    :param trg:  target tensor size of (n_batch x max_seq_len x target_size)
    :param trg_len: target sequence sizes
    :return: loss value over batch elements
    """
    loss = torch.zeros(1, device=pred.device)
    lossfn = nn.BCELoss()

    while trg.dim() < pred.dim():
        trg = trg.unsqueeze(0)

    for a_batch_instance in zip(pred, trg, trg_len):
        probs_b, target_b, len_b = a_batch_instance

        if len_b == 1:
            probs = probs_b
            trgs = target_b
        else:
            probs = probs_b[:len_b]
            trgs = target_b[:len_b]

        if trgs.numel() == 1:
            continue

        probs = probs 
        trgs = trgs.to(probs.device)
        try:
            loss += lossfn(probs, trgs)
        except RuntimeError as e:
            raise e
        except ValueError as e:
            raise e

    return loss / len(trg_len)


class BaseMultiLabelLSTM(nn.Module):
    """
    input: multi-hot vectored sequence input
    output: multi-hot vectored sequence output
    """
    def __init__(self, event_size, window_size_y, target_size, hidden_dim, embed_dim,
                 use_cuda=False, batch_size=64, batch_first=True,
                 bidirectional=False, 
                 rnn_type='GRU', num_layers=1, 
                 padding_idx=0,
                 device=None,
                 is_input_time=False,
                 dropout=0,
                 dropouti=0,
                 dropouth=0,
                 wdrop=0,
                 remap=False,
                 f_window=False,
                 manual_alpha=0,
                 inv_id_mapping=None,
                 elapsed_time=False,
                 ):
        super(BaseMultiLabelLSTM, self).__init__()

        self.event_size = event_size
        self.target_size = target_size
        self.hidden_dim = hidden_dim
        self.embedding_dim = embed_dim
        self.use_cuda = use_cuda
        self.batch_size = batch_size
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.num_layers = num_layers

        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        self.dropout_layer_input = nn.Dropout(dropouti)
        self.rnn_type = rnn_type
        self.device = device
        self.padding_idx = padding_idx

        self.drop = nn.Dropout(dropout)
        self.inv_id_mapping = inv_id_mapping
    
        if rnn_type == 'GRU':
            self.is_pack_seq = True
        else:
            self.is_pack_seq = False

        self.fc_out = nn.Linear(hidden_dim, target_size)

        self.embed_input = nn.Linear(event_size, embed_dim, bias=False)
            
        self.init_weights(padding_idx=padding_idx)

        if self.rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size=embed_dim,
                                     hidden_size=hidden_dim,
                                     batch_first=batch_first,
                                     bidirectional=bidirectional,
                                     num_layers=num_layers,
                                     dropout=dropout)

        elif self.rnn_type == 'retain':
            self.rnn = RETAIN(dim_input=event_size, dim_emb=embed_dim, 
                dropout_input=self.dropout, dropout_emb=self.dropout, 
                dim_alpha=1, dim_beta=hidden_dim,
                dropout_context=self.dropout, dim_output=hidden_dim, batch_first=True)

        elif self.rnn_type == 'CNN':
            # kernel determines how large days we apply conv1d operation 
            self.cnn_kernels = [2,4,8]
            self.convs = nn.ModuleDict({f"{i}": nn.ModuleList([nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=k).to(device) \
                    for k in self.cnn_kernels]) for i in range(self.num_layers)})
            self.pooling = nn.AdaptiveMaxPool1d(1)
            n_channels = len(self.cnn_kernels)
            self.fc_cnn = nn.Linear(n_channels * embed_dim, target_size)


    def forward(self, seq_events, lengths, hidden):
        """
        inputs: n_batch x seqlen x 1
        """

        # Get Embedding and prep sequence
    
        seq_events = seq_events.float()

        if self.rnn_type not in ['retain']:
            input_seq = self.embed_input(seq_events)
            # embed : n_batch x seqlen x 1
            input_seq = self.dropout_layer_input(input_seq)
        else:
            input_seq = seq_events
        
        if self.rnn_type == 'GRU':
            input_seq = pack_padded_sequence(input_seq, lengths,
                                             batch_first=True)
        # Run RNN
        if self.rnn_type  == 'GRU':
            _output, hidden = self.rnn(input_seq, hidden)
            _output, output_lengths = pad_packed_sequence(_output,
                                                     batch_first=self.batch_first,
                                                     total_length=max(lengths))
        elif self.rnn_type == 'retain':
            _output = []
            for step in range(1, seq_events.size(1) + 1):
                seq_step = seq_events[:, :step]
                length_step = [min(x, step) for x in lengths]
                _output_step, alphas, betas = self.rnn(seq_step, length_step)
                _output.append(_output_step)
            
            _output = torch.stack(_output, dim=1)


        if self.rnn_type == "CNN":
            # based on http://www.cse.chalmers.se/~richajo/nlp2019/l2/Text%20classification%20using%20a%20word-based%20CNN.html
            # permute input dim (batch, seq_len, embed_dim) to (batch, embed_dim, seq_len)
            # to make sure that we convolve over the last dimension (seq_len)
            preds = []

            for i in range(input_seq.size(1)):
                
                step_seq = input_seq[:, :i + 1]


                for idx, conv_layer in enumerate(self.convs.values()):

                    nb, ns, ne = step_seq.size()
                    # add padding for shorter sequence than max kernel
                    if ns < max(self.cnn_kernels):
                        len_need = max(self.cnn_kernels) - ns
                        pad = torch.zeros(nb, len_need, ne).to(self.device)
                        step_seq = torch.cat((pad, step_seq), dim=1)

                    x = step_seq.permute(0, 2, 1)
                    
                    conv_maps = [torch.relu(conv(x)) for conv in conv_layer]
                    pooled = [self.pooling(conv_map) for conv_map in conv_maps]
                    # merge all different conv kernels into single one
                    all_pooled = torch.cat(pooled, 1)
                    all_pooled = all_pooled.squeeze(2)
                    step_seq = self.dropout_layer(all_pooled)
                    if idx < self.num_layers - 1:
                        step_seq = step_seq.reshape(nb, -1, ne)

                pred_step = self.fc_cnn(step_seq)
                preds.append(pred_step)
            plain_output = torch.stack(preds, dim=1)

        if self.rnn_type != "CNN":
            # dropout on output of RNN/LSTM output hidden states
            _output = self.dropout_layer(_output)

            plain_output = self.fc_out(_output)

        assert (plain_output != plain_output).sum() == 0, "NaN!"

        return plain_output, hidden

    def init_hidden(self, batch_size=None, device=None):
        init = 0.1
        if device is None:
            device = self.device
            
        if not batch_size:
            batch_size = self.batch_size
        h0 = torch.randn(self.num_layers * self.num_directions,
                        batch_size, self.hidden_dim)
        c0 = torch.randn(self.num_layers * self.num_directions,
                        batch_size, self.hidden_dim)
        h0.data.uniform_(-init, init)
        c0.data.uniform_(-init, init)
        hx = [h0, c0]
        hx = [x.to(device) for x in hx]
        
        if self.rnn_type in ['GRU']:
            hx = hx[0]
        return hx

    def init_weights(self, padding_idx=None):
        init = 0.1
        if self.embed_input is not None:
            self.embed_input.weight.data.uniform_(-init, init)

        if padding_idx:
            if self.embed_input is not None:
                self.embed_input.weight.data[padding_idx] = 0

