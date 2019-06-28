import logging
import six

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from espnet.nets.e2e_asr_common import get_vgg2l_odim
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.nets_utils import to_device


class RNNP(torch.nn.Module):
    """RNN with projection layer module

    :param int idim: dimension of inputs
    :param int elayers: number of encoder layers
    :param int cdim: number of rnn units (resulted in cdim * 2 if bidirectional)
    :param int hdim: number of projection units
    :param np.ndarray subsample: list of subsampling numbers
    :param float dropout: dropout rate
    :param str typ: The RNN type
    """

    def __init__(self, idim, elayers, cdim, hdim, subsample, dropout, typ="blstm"):
        super(RNNP, self).__init__()
        bidir = typ[0] == "b"
        for i in six.moves.range(elayers):
            if i == 0:
                inputdim = idim
            else:
                inputdim = hdim
            rnn = torch.nn.LSTM(inputdim, cdim, dropout=dropout, num_layers=1, bidirectional=bidir,
                                batch_first=True) if "lstm" in typ \
                else torch.nn.GRU(inputdim, cdim, dropout=dropout, num_layers=1, bidirectional=bidir, batch_first=True)
            setattr(self, "%s%d" % ("birnn" if bidir else "rnn", i), rnn)
            # bottleneck layer to merge
            if bidir:
                setattr(self, "bt%d" % i, torch.nn.Linear(2 * cdim, hdim))
            else:
                setattr(self, "bt%d" % i, torch.nn.Linear(cdim, hdim))

        self.elayers = elayers
        self.cdim = cdim
        self.subsample = subsample
        self.typ = typ
        self.bidir = bidir

    def forward(self, xs_pad, ilens, prev_state=None):
        """RNNP forward

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor prev_state: batch of previous RNN states
        :return: batch of hidden state sequences (B, Tmax, hdim)
        :rtype: torch.Tensor
        """
        # logging.info(self.__class__.__name__ + ' input lengths: ' + str(ilens))
        elayer_states = []
        for layer in six.moves.range(self.elayers):
            xs_pack = pack_padded_sequence(xs_pad, ilens, batch_first=True)
            rnn = getattr(self, ("birnn" if self.bidir else "rnn") + str(layer))
            rnn.flatten_parameters()
            if prev_state is not None and rnn.bidirectional:
                prev_state = reset_backward_rnn_state(prev_state)
            ys, states = rnn(xs_pack, hx=prev_state)
            elayer_states.append(states)
            # ys: utt list of frame x cdim x 2 (2: means bidirectional)
            ys_pad, ilens = pad_packed_sequence(ys, batch_first=True)
            sub = self.subsample[layer + 1]
            if sub > 1:
                ys_pad = ys_pad[:, ::sub]
                ilens = [int(i + 1) // sub for i in ilens]
            # (sum _utt frame_utt) x dim
            projected = getattr(self, 'bt' + str(layer)
                                )(ys_pad.contiguous().view(-1, ys_pad.size(2)))
            xs_pad = torch.tanh(projected.view(ys_pad.size(0), ys_pad.size(1), -1))

        return xs_pad, ilens, elayer_states  # x: utt list of frame x dim


class RNN(torch.nn.Module):
    """RNN module

    :param int idim: dimension of inputs
    :param int elayers: number of encoder layers
    :param int cdim: number of rnn units (resulted in cdim * 2 if bidirectional)
    :param int hdim: number of final projection units
    :param float dropout: dropout rate
    :param str typ: The RNN type
    """

    def __init__(self, idim, elayers, cdim, hdim, dropout, typ="blstm"):
        super(RNN, self).__init__()
        bidir = typ[0] == "b"
        self.nbrnn = torch.nn.LSTM(idim, cdim, elayers, batch_first=True,
                                   dropout=dropout, bidirectional=bidir) if "lstm" in typ \
            else torch.nn.GRU(idim, cdim, elayers, batch_first=True, dropout=dropout,
                              bidirectional=bidir)
        if bidir:
            self.l_last = torch.nn.Linear(cdim * 2, hdim)
        else:
            self.l_last = torch.nn.Linear(cdim, hdim)
        self.typ = typ

    def forward(self, xs_pad, ilens, prev_state=None):
        """RNN forward

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, D)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor prev_state: batch of previous RNN states
        :return: batch of hidden state sequences (B, Tmax, eprojs)
        :rtype: torch.Tensor
        """
        logging.info(self.__class__.__name__ + ' input lengths: ' + str(ilens))
        xs_pack = pack_padded_sequence(xs_pad, ilens, batch_first=True)
        self.nbrnn.flatten_parameters()
        if prev_state is not None and self.nbrnn.bidirectional:
            # We assume that when previous state is passed, it means that we're streaming the input
            # and therefore cannot propagate backward BRNN state (otherwise it goes in the wrong direction)
            prev_state = reset_backward_rnn_state(prev_state)
        ys, states = self.nbrnn(xs_pack, hx=prev_state)
        # ys: utt list of frame x cdim x 2 (2: means bidirectional)
        ys_pad, ilens = pad_packed_sequence(ys, batch_first=True)
        # (sum _utt frame_utt) x dim
        projected = torch.tanh(self.l_last(
            ys_pad.contiguous().view(-1, ys_pad.size(2))))
        xs_pad = projected.view(ys_pad.size(0), ys_pad.size(1), -1)
        return xs_pad, ilens, states  # x: utt list of frame x dim


def reset_backward_rnn_state(states):
    """Sets backward BRNN states to zeroes - useful in processing of sliding windows over the inputs"""
    if isinstance(states, (list, tuple)):
        for state in states:
            state[1::2] = 0.
    else:
        states[1::2] = 0.
    return states


class VGG2L(torch.nn.Module):
    """VGG-like module

    :param int in_channel: number of input channels
    """

    def __init__(self, in_channel=1):
        super(VGG2L, self).__init__()
        # CNN layer (VGG motivated)
        self.conv1_1 = torch.nn.Conv2d(in_channel, 64, 3, stride=1, padding=1)
        self.conv1_2 = torch.nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv2_1 = torch.nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv2_2 = torch.nn.Conv2d(128, 128, 3, stride=1, padding=1)

        self.in_channel = in_channel

    def forward(self, xs_pad, ilens, **kwargs):
        """VGG2L forward

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, D)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :return: batch of padded hidden state sequences (B, Tmax // 4, 128)
        :rtype: torch.Tensor
        """
        logging.info(self.__class__.__name__ + ' input lengths: ' + str(ilens))

        # x: utt x frame x dim
        # xs_pad = F.pad_sequence(xs_pad)

        # x: utt x 1 (input channel num) x frame x dim
        xs_pad = xs_pad.view(xs_pad.size(0), xs_pad.size(1), self.in_channel,
                             xs_pad.size(2) // self.in_channel).transpose(1, 2)

        # NOTE: max_pool1d ?
        xs_pad = F.relu(self.conv1_1(xs_pad))
        xs_pad = F.relu(self.conv1_2(xs_pad))
        xs_pad = F.max_pool2d(xs_pad, 2, stride=2, ceil_mode=True)

        xs_pad = F.relu(self.conv2_1(xs_pad))
        xs_pad = F.relu(self.conv2_2(xs_pad))
        xs_pad = F.max_pool2d(xs_pad, 2, stride=2, ceil_mode=True)
        if torch.is_tensor(ilens):
            ilens = ilens.cpu().numpy()
        else:
            ilens = np.array(ilens, dtype=np.float32)
        ilens = np.array(np.ceil(ilens / 2), dtype=np.int64)
        ilens = np.array(
            np.ceil(np.array(ilens, dtype=np.float32) / 2), dtype=np.int64).tolist()

        # x: utt_list of frame (remove zeropaded frames) x (input channel num x dim)
        xs_pad = xs_pad.transpose(1, 2)
        xs_pad = xs_pad.contiguous().view(
            xs_pad.size(0), xs_pad.size(1), xs_pad.size(2) * xs_pad.size(3))
        return xs_pad, ilens, None  # no state in this layer


class Encoder(torch.nn.Module):
    """Encoder module

    :param str etype: type of encoder network
    :param int idim: number of dimensions of encoder network
    :param int elayers: number of layers of encoder network
    :param int eunits: number of lstm units of encoder network
    :param int eprojs: number of projection units of encoder network
    :param np.ndarray subsample: list of subsampling numbers
    :param float dropout: dropout rate
    :param int in_channel: number of input channels
    """

    def __init__(self, etype, idim, elayers, eunits, eprojs, subsample, dropout, in_channel=1):
        super(Encoder, self).__init__()
        typ = etype.lstrip("vgg").rstrip("p")
        if typ not in ['lstm', 'gru', 'blstm', 'bgru']:
            logging.error("Error: need to specify an appropriate encoder architecture")

        if etype.startswith("vgg"):
            if etype[-1] == "p":
                self.enc = torch.nn.ModuleList([VGG2L(in_channel),
                                                RNNP(get_vgg2l_odim(idim, in_channel=in_channel), elayers, eunits,
                                                     eprojs,
                                                     subsample, dropout, typ=typ)])
                logging.info('Use CNN-VGG + ' + typ.upper() + 'P for encoder')
            else:
                self.enc = torch.nn.ModuleList([VGG2L(in_channel),
                                                RNN(get_vgg2l_odim(idim, in_channel=in_channel), elayers, eunits,
                                                    eprojs,
                                                    dropout, typ=typ)])
                logging.info('Use CNN-VGG + ' + typ.upper() + ' for encoder')
        else:
            if etype[-1] == "p":
                self.enc = torch.nn.ModuleList(
                    [RNNP(idim, elayers, eunits, eprojs, subsample, dropout, typ=typ)])
                logging.info(typ.upper() + ' with every-layer projection for encoder')
            else:
                self.enc = torch.nn.ModuleList([RNN(idim, elayers, eunits, eprojs, dropout, typ=typ)])
                logging.info(typ.upper() + ' without projection for encoder')

    def forward(self, xs_pad, ilens, prev_states=None):
        """Encoder forward

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, D)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor prev_state: batch of previous encoder hidden states (?, ...)
        :return: batch of hidden state sequences (B, Tmax, eprojs)
        :rtype: torch.Tensor
        """
        if prev_states is None:
            prev_states = [None] * len(self.enc)
        assert len(prev_states) == len(self.enc)

        current_states = []
        for module, prev_state in zip(self.enc, prev_states):
            xs_pad, ilens, states = module(xs_pad, ilens, prev_state=prev_state)
            current_states.append(states)

        # make mask to remove bias value in padded part
        mask = to_device(self, make_pad_mask(ilens).unsqueeze(-1))

        return xs_pad.masked_fill(mask, 0.0), ilens, current_states


class MultiLevelEncoder(torch.nn.Module):
    """Encoder module

    :param str etype: type of encoder network
    :param int idim: number of dimensions of encoder network
    :param int elayers: number of layers of encoder network
    :param int eunits: number of lstm units of encoder network
    :param int eprojs: number of projection units of encoder network
    :param np.ndarray subsample: list of subsampling numbers
    :param float dropout: dropout rate
    :param int in_channel: number of input channels
    """

    def __init__(self, etype, idim, elayers, eunits, eprojs, subsample, dropout, gunits, gprojs, in_channel=1):
        super(MultiLevelEncoder, self).__init__()
        typ = etype.lstrip("vgg")
        if typ not in ['blstm']:
            logging.error("Error: need to specify an appropriate encoder architecture")

        self.elayers = elayers
        self.vgg = VGG2L(in_channel)
        self.nbrnn = torch.nn.LSTM(get_vgg2l_odim(idim, in_channel=in_channel), eunits, elayers - 1, batch_first=True,
                                   dropout=dropout, bidirectional=True)

        self.ml_brnn_1 = torch.nn.LSTM(2 * eunits, eunits, 1, batch_first=True,
                                   dropout=0.0, bidirectional=True)

        self.dropout_1 = torch.nn.Dropout(p=dropout)

        self.ml_brnn_2 = torch.nn.LSTM(2 * eunits, gunits, 1, batch_first=True,
                                       dropout=0.0, bidirectional=True)

        self.l_last_1 = torch.nn.Linear(eunits * 2, eprojs)

        self.l_last_2 = torch.nn.Linear(gunits * 2, gprojs)

        logging.info('MultiLevelEncoder with VGG for encoder')

    def forward(self, xs_pad, ilens, prev_states=None):
        """Encoder forward

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, D)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor prev_state: batch of previous encoder hidden states (?, ...)
        :return: batch of hidden state sequences (B, Tmax, eprojs)
        :rtype: torch.Tensor
        """
        xs_pad, ilens, states = self.vgg(xs_pad, ilens)
        xs_pad = self.dropout_1(xs_pad)

        current_states = []

        logging.info(self.__class__.__name__ + ' input lengths: ' + str(ilens))
        xs_pack = pack_padded_sequence(xs_pad, ilens, batch_first=True)
        self.nbrnn.flatten_parameters()
        self.ml_brnn_1.flatten_parameters()
        self.ml_brnn_2.flatten_parameters()

        ys, states = self.nbrnn(xs_pack)
        current_states.append(states)

        ys1, states1 = self.ml_brnn_1(ys)  # Adds dropout after the common LSTM layers
        ys2, states2 = self.ml_brnn_2(ys)  # Adds dropout after the common LSTM layers

        current_states.append(states1)
        current_states.append(states2)
        # ys: utt list of frame x cdim x 2 (2: means bidirectional)
        ys_pad1, ilens1 = pad_packed_sequence(ys1, batch_first=True)
        ys_pad2, ilens2 = pad_packed_sequence(ys2, batch_first=True)
        # (sum _utt frame_utt) x dim
        projected1 = torch.tanh(self.l_last_1(
            ys_pad1.contiguous().view(-1, ys_pad1.size(2))))
        xs_pad1 = projected1.view(ys_pad1.size(0), ys_pad1.size(1), -1)

        projected2 = torch.tanh(self.l_last_2(
            ys_pad2.contiguous().view(-1, ys_pad2.size(2))))
        xs_pad2 = projected2.view(ys_pad2.size(0), ys_pad2.size(1), -1)

        # make mask to remove bias value in padded part
        mask = to_device(self, make_pad_mask(ilens).unsqueeze(-1))

        return xs_pad1.masked_fill(mask, 0.0), xs_pad2.masked_fill(mask, 0.0), ilens1, states


class MultiLayerGlobalAttentionEncoder(torch.nn.Module):
    def __init__(self, in_dim, hid, in_channel=1):
        super(MultiLayerGlobalAttentionEncoder, self).__init__()

        self.gproj = 128
        self.hidden_dim = 2 * hid
        self.vgg = VGG2L(in_channel)
        self.dropout_cnn = torch.nn.Dropout(0.2)

        self.cnnout_dim = get_vgg2l_odim(in_dim, in_channel=in_channel)
        self.cnn_proj = torch.nn.Linear(self.cnnout_dim, 2 * hid)  # 2688 x 128

        self.gatt_scale = 0.7
        self.h_length = None

        self.layer_1 = torch.nn.LSTM(2 * hid + self.gproj, hid, 1, batch_first=True, bidirectional=True)
        self.layer_2 = torch.nn.LSTM(2 * hid + self.gproj, hid, 1, batch_first=True, bidirectional=True)
        self.layer_3 = torch.nn.LSTM(2 * hid + self.gproj, hid, 1, batch_first=True, bidirectional=True)

        self.l_last = torch.nn.Linear(2 * hid + self.gproj, hid)

        self.dropout_1 = torch.nn.Dropout(0.2)
        self.dropout_2 = torch.nn.Dropout(0.2)

        self.global_attention_mlp_01 = torch.nn.Linear(2 * hid, hid, bias=False)
        self.global_attention_mlp_02 = torch.nn.Linear(hid, 1, bias=False)
        self.global_attention_mlp_03 = torch.nn.Linear(2 * hid, self.gproj, bias=False)

        self.global_attention_mlp_11 = torch.nn.Linear(2 * hid, hid, bias=False)
        self.global_attention_mlp_12 = torch.nn.Linear(hid, 1, bias=False)
        self.global_attention_mlp_13 = torch.nn.Linear(2 * hid, self.gproj, bias=False)

        self.global_attention_mlp_21 = torch.nn.Linear(2 * hid, hid, bias=False)
        self.global_attention_mlp_22 = torch.nn.Linear(hid, 1, bias=False)
        self.global_attention_mlp_23 = torch.nn.Linear(2 * hid, self.gproj, bias=False)

        self.global_attention_mlp_31 = torch.nn.Linear(2 * hid, hid, bias=False)
        self.global_attention_mlp_32 = torch.nn.Linear(hid, 1, bias=False)
        self.global_attention_mlp_33 = torch.nn.Linear(2 * hid, self.gproj, bias=False)

        self.g_adapt_0 = torch.nn.Parameter(torch.zeros(1, 2 * hid, requires_grad=False))
        self.g_adapt_1 = torch.nn.Parameter(torch.zeros(1, 2 * hid, requires_grad=False))
        self.g_adapt_2 = torch.nn.Parameter(torch.zeros(1, 2 * hid, requires_grad=False))
        self.g_adapt_3 = torch.nn.Parameter(torch.zeros(1, 2 * hid, requires_grad=False))

        self.weight_0 = torch.nn.Parameter(torch.Tensor([1.0]))
        self.weight_1 = torch.nn.Parameter(torch.Tensor([1.0]))
        self.weight_2 = torch.nn.Parameter(torch.Tensor([1.0]))
        self.weight_3 = torch.nn.Parameter(torch.Tensor([1.0]))

    def forward(self, xs_pad, ilens):
        xs_pad, ilens, states = self.vgg(xs_pad, ilens)

        xs_pad = self.dropout_cnn(xs_pad)
        xs_pad = self.cnn_proj(xs_pad)

        self.h_length = xs_pad.size(1)
        batch = len(ilens)

        self.mask = to_device(self, make_pad_mask(ilens))

        gt_01 = torch.tanh(self.global_attention_mlp_01(xs_pad))
        gt_02 = self.global_attention_mlp_02(gt_01).squeeze(2)
        # NOTE consider zero padding when compute gt_02.
        gt_02.masked_fill_(self.mask, -float('inf'))
        gw0 = F.softmax(self.gatt_scale * gt_02, dim=1)
        ga0 = self.g_adapt_0 + torch.sum(xs_pad * gw0.view(batch, self.h_length, 1), dim=1)
        ga0_projected = self.global_attention_mlp_03(ga0)
        ga0_projected = ga0_projected.view(batch, 1, -1)
        ga0_projected_exp = ga0_projected.expand(batch, self.h_length, self.gproj)
        xs_pad = torch.cat((xs_pad, ga0_projected_exp), dim=2)

        self.layer_1.flatten_parameters()
        self.layer_2.flatten_parameters()
        self.layer_3.flatten_parameters()

        xs_pack = pack_padded_sequence(xs_pad, ilens, batch_first=True)

        # first BLSTM Layer
        h1, _ = self.layer_1(xs_pack)
        h1, ilens = pad_packed_sequence(h1, batch_first=True)
        h1 = self.dropout_1(h1)  # Take the global attention over this.

        gt_11 = torch.tanh(self.global_attention_mlp_11(h1))
        gt_12 = self.global_attention_mlp_12(gt_11).squeeze(2)
        # NOTE consider zero padding when compute gt_12.
        gt_12.masked_fill_(self.mask, -float('inf'))
        gw1 = F.softmax(self.gatt_scale * gt_12, dim=1)
        ga1 = self.g_adapt_1 + torch.sum(h1 * gw1.view(batch, self.h_length, 1), dim=1)

        ga1_projected = self.global_attention_mlp_13(ga1)
        ga1_projected = ga1_projected.view(batch, 1, -1)
        ga1_projected_exp = ga1_projected.expand(batch, self.h_length, self.gproj)
        h1 = torch.cat((h1, ga1_projected_exp), dim=2)

        xs_pack = pack_padded_sequence(h1, ilens, batch_first=True)

        # Second BLSTM Layer
        h2, _ = self.layer_2(xs_pack)
        h2, ilens = pad_packed_sequence(h2, batch_first=True)
        h2 = self.dropout_2(h2)  # Take the global attention over this.

        gt_21 = torch.tanh(self.global_attention_mlp_21(h2))
        gt_22 = self.global_attention_mlp_22(gt_21).squeeze(2)
        # NOTE consider zero padding when compute gt_12.
        gt_22.masked_fill_(self.mask, -float('inf'))
        gw2 = F.softmax(self.gatt_scale * gt_22, dim=1)
        ga2 = self.g_adapt_2 + torch.sum(h2 * gw2.view(batch, self.h_length, 1), dim=1)

        ga2_projected = self.global_attention_mlp_23(ga2)
        ga2_projected = ga2_projected.view(batch, 1, -1)
        ga2_projected_exp = ga2_projected.expand(batch, self.h_length, self.gproj)
        h2 = torch.cat((h2, ga2_projected_exp), dim=2)

        xs_pack = pack_padded_sequence(h2, ilens, batch_first=True)

        # Third BLSTM Layer
        h3, _ = self.layer_3(xs_pack)
        h3, ilens = pad_packed_sequence(h3, batch_first=True)
        #         h3 = self.dropout_3(h3)   #Take the global attention over this.

        gt_31 = torch.tanh(self.global_attention_mlp_31(h3))
        gt_32 = self.global_attention_mlp_32(gt_31).squeeze(2)
        # NOTE consider zero padding when compute gt_12.
        gt_32.masked_fill_(self.mask, -float('inf'))
        gw3 = F.softmax(self.gatt_scale * gt_32, dim=1)
        ga3 = self.g_adapt_3 + torch.sum(h3 * gw3.view(batch, self.h_length, 1), dim=1)

        ga3_projected = self.global_attention_mlp_33(ga3)
        ga3_projected = ga3_projected.view(batch, 1, -1)
        ga3_projected_exp = ga3_projected.expand(batch, self.h_length, self.gproj)
        h3 = torch.cat((h3, ga3_projected_exp), dim=2)

        global_attention = self.weight_0 * ga0_projected + self.weight_1 * ga1_projected + self.weight_2 * ga2_projected + self.weight_3 * ga3_projected
        ga_projected = torch.tanh(global_attention.contiguous())

        projected = torch.tanh(self.l_last(
            h3.contiguous().view(-1, h3.size(2))))
        projected = projected.view(h3.size(0), h3.size(1), -1)

        return projected, ilens, ga_projected.squeeze(1)

# class MultiLayerGlobalAttentionEncoder(torch.nn.Module):
#     def __init__(self, in_dim, hid, in_channel=1):
#         super(MultiLayerGlobalAttentionEncoder, self).__init__()
#
#         self.vgg = VGG2L(in_channel)
#         self.dropout_cnn = torch.nn.Dropout(0.2)
#
#         self.cnnout_dim = get_vgg2l_odim(in_dim, in_channel=in_channel)
#         self.cnn_proj = torch.nn.Linear(self.cnnout_dim, 2 * hid)
#
#         self.gatt_scale = 0.7
#         self.h_length = None
#
#         self.layer_1 = torch.nn.LSTM(2 * hid, hid, 1, batch_first=True,
#                                      bidirectional=True)
#         self.layer_2 = torch.nn.LSTM(2 * hid, hid, 1, batch_first=True, bidirectional=True)
#         self.layer_3 = torch.nn.LSTM(2 * hid, hid, 1, batch_first=True, bidirectional=True)
#
#         self.l_last = torch.nn.Linear(2 * hid, hid)
#         self.l_gt_last = torch.nn.Linear(2 * hid, hid)
#
#         self.dropout_1 = torch.nn.Dropout(0.2)
#         self.dropout_2 = torch.nn.Dropout(0.2)
#
#         self.global_attention_mlp_01 = torch.nn.Linear(2 * hid, hid, bias=False)
#         self.global_attention_mlp_02 = torch.nn.Linear(hid, 1, bias=False)
#
#         self.global_attention_mlp_11 = torch.nn.Linear(2 * hid, hid, bias=False)
#         self.global_attention_mlp_12 = torch.nn.Linear(hid, 1, bias=False)
#
#         self.global_attention_mlp_21 = torch.nn.Linear(2 * hid, hid, bias=False)
#         self.global_attention_mlp_22 = torch.nn.Linear(hid, 1, bias=False)
#
#         self.global_attention_mlp_31 = torch.nn.Linear(2 * hid, hid, bias=False)
#         self.global_attention_mlp_32 = torch.nn.Linear(hid, 1, bias=False)
#
#         self.g_adapt_0 = torch.nn.Parameter(torch.zeros(1, 2 * hid, requires_grad=False))
#         self.g_adapt_1 = torch.nn.Parameter(torch.zeros(1, 2 * hid, requires_grad=False))
#         self.g_adapt_2 = torch.nn.Parameter(torch.zeros(1, 2 * hid, requires_grad=False))
#         self.g_adapt_3 = torch.nn.Parameter(torch.zeros(1, 2 * hid, requires_grad=False))
#
#         self.weight_0 = torch.nn.Parameter(torch.Tensor([0.25]))
#         self.weight_1 = torch.nn.Parameter(torch.Tensor([0.25]))
#         self.weight_2 = torch.nn.Parameter(torch.Tensor([0.25]))
#         self.weight_3 = torch.nn.Parameter(torch.Tensor([0.25]))
#
#     def forward(self, xs_pad, ilens):
#         xs_pad, ilens, states = self.vgg(xs_pad, ilens)
#         xs_pad = self.dropout_cnn(xs_pad)
#         xs_pad = self.cnn_proj(xs_pad)
#
#         self.h_length = xs_pad.size(1)
#         batch = len(ilens)
#
#         self.mask = to_device(self, make_pad_mask(ilens))
#
#         gt_01 = torch.tanh(self.global_attention_mlp_01(xs_pad))
#         gt_02 = self.global_attention_mlp_02(gt_01).squeeze(2)
#         # NOTE consider zero padding when compute gt_02.
#         gt_02.masked_fill_(self.mask, -float('inf'))
#         gw0 = F.softmax(self.gatt_scale * gt_02, dim=1)
#         ga0 = self.g_adapt_0 + torch.sum(xs_pad * gw0.view(batch, self.h_length, 1), dim=1)
#
#         xs_pad = ga0.view(batch, 1, -1) + xs_pad
#
#         self.layer_1.flatten_parameters()
#         self.layer_2.flatten_parameters()
#         self.layer_3.flatten_parameters()
#
#         xs_pack = pack_padded_sequence(xs_pad, ilens, batch_first=True)
#
#         # first BLSTM Layer
#         h1, _ = self.layer_1(xs_pack)
#         h1, ilens = pad_packed_sequence(h1, batch_first=True)
#         h1 = self.dropout_1(h1)  # Take the global attention over this.
#
#         gt_11 = torch.tanh(self.global_attention_mlp_11(h1))
#         gt_12 = self.global_attention_mlp_12(gt_11).squeeze(2)
#         # NOTE consider zero padding when compute gt_12.
#         gt_12.masked_fill_(self.mask, -float('inf'))
#         gw1 = F.softmax(self.gatt_scale * gt_12, dim=1)
#         ga1 = self.g_adapt_1 + torch.sum(h1 * gw1.view(batch, self.h_length, 1), dim=1)
#
#         h1 = ga1.view(batch, 1, -1) + h1
#
#         xs_pack = pack_padded_sequence(h1, ilens, batch_first=True)
#
#         # Second BLSTM Layer
#         h2, _ = self.layer_2(xs_pack)
#         h2, ilens = pad_packed_sequence(h2, batch_first=True)
#         h2 = self.dropout_2(h2)  # Take the global attention over this.
#
#         gt_21 = torch.tanh(self.global_attention_mlp_21(h2))
#         gt_22 = self.global_attention_mlp_22(gt_21).squeeze(2)
#         # NOTE consider zero padding when compute gt_12.
#         gt_22.masked_fill_(self.mask, -float('inf'))
#         gw2 = F.softmax(self.gatt_scale * gt_22, dim=1)
#         ga2 = self.g_adapt_2 + torch.sum(h2 * gw2.view(batch, self.h_length, 1), dim=1)
#
#         h2 = ga2.view(batch, 1, -1) + h2
#
#         xs_pack = pack_padded_sequence(h2, ilens, batch_first=True)
#
#         # Third BLSTM Layer
#         h3, _ = self.layer_3(xs_pack)
#         h3, ilens = pad_packed_sequence(h3, batch_first=True)
#         #         h3 = self.dropout_3(h3)   #Take the global attention over this.
#
#         gt_31 = torch.tanh(self.global_attention_mlp_31(h3))
#         gt_32 = self.global_attention_mlp_32(gt_31).squeeze(2)
#         # NOTE consider zero padding when compute gt_12.
#         # gt_32.masked_fill_(self.mask, -float('inf'))
#         gw3 = F.softmax(self.gatt_scale * gt_32, dim=1)
#         ga3 = self.g_adapt_3 + torch.sum(h3 * gw3.view(batch, self.h_length, 1), dim=1)
#
#         h3 = ga3.view(batch, 1, -1) + h3
#
#         global_attention = self.weight_0 * ga0 + self.weight_1 * ga1 + self.weight_2 * ga2 + self.weight_3 * ga3
#         #         global_attention = self.weight_1 * ga1 + self.weight_2 * ga2 + self.weight_3 * ga3 + self.weight_4 * ga4
#         ga_projected = torch.tanh(self.l_gt_last(
#             global_attention.contiguous()))
#         # ga_projected = ga_projected.view(global_attention.size(0), global_attention.size(1), -1)
#
#         projected = torch.tanh(self.l_last(
#             h3.contiguous().view(-1, h3.size(2))))
#         projected = projected.view(h3.size(0), h3.size(1), -1)
#
#         return projected, ilens, ga_projected


# class MultiLayerGlobalAttentionEncoder(torch.nn.Module):
#     def __init__(self, in_dim, hid, in_channel=1):
#         super(MultiLayerGlobalAttentionEncoder, self).__init__()
#
#         self.vgg = VGG2L(in_channel)
#         self.dropout_cnn = torch.nn.Dropout(0.2)
#
#         self.cnnout_dim = get_vgg2l_odim(in_dim, in_channel=in_channel)
#         self.cnn_proj = torch.nn.Linear(self.cnnout_dim, 2 * hid)
#
#         self.gatt_scale = 0.7
#         self.h_length = None
#         self.mask = None
#
#         self.layer_1 = torch.nn.LSTM(2 * hid, hid, 1, batch_first=True,
#                                      bidirectional=True)
#         self.layer_2 = torch.nn.LSTM(2 * hid, hid, 1, batch_first=True, bidirectional=True)
#         self.layer_3 = torch.nn.LSTM(2 * hid, hid, 1, batch_first=True, bidirectional=True)
#
#         self.l_last = torch.nn.Linear(2 * hid, hid)
#         self.l_gt_last = torch.nn.Linear(2 * hid, hid)
#
#         self.dropout_1 = torch.nn.Dropout(0.2)
#         self.dropout_2 = torch.nn.Dropout(0.2)
#
#         self.global_attention_mlp_01 = torch.nn.Linear(2 * hid, hid, bias=False)
#         self.global_attention_mlp_02 = torch.nn.Linear(hid, 1, bias=False)
#
#         self.global_attention_mlp_11 = torch.nn.Linear(2 * hid, hid, bias=False)
#         self.global_attention_mlp_12 = torch.nn.Linear(hid, 1, bias=False)
#
#         self.global_attention_mlp_21 = torch.nn.Linear(2 * hid, hid, bias=False)
#         self.global_attention_mlp_22 = torch.nn.Linear(hid, 1, bias=False)
#
#         self.global_attention_mlp_31 = torch.nn.Linear(2 * hid, hid, bias=False)
#         self.global_attention_mlp_32 = torch.nn.Linear(hid, 1, bias=False)
#
#         self.g_adapt_0 = torch.nn.Parameter(torch.zeros(1, 2 * hid, requires_grad=False))
#         self.g_adapt_1 = torch.nn.Parameter(torch.zeros(1, 2 * hid, requires_grad=False))
#         self.g_adapt_2 = torch.nn.Parameter(torch.zeros(1, 2 * hid, requires_grad=False))
#         self.g_adapt_3 = torch.nn.Parameter(torch.zeros(1, 2 * hid, requires_grad=False))
#
#         self.weight_0 = torch.nn.Parameter(torch.Tensor([0.25]))
#         self.weight_1 = torch.nn.Parameter(torch.Tensor([0.25]))
#         self.weight_2 = torch.nn.Parameter(torch.Tensor([0.25]))
#         self.weight_3 = torch.nn.Parameter(torch.Tensor([0.25]))
#
#     def forward(self, xs_pad, ilens):
#         xs_pad, ilens, states = self.vgg(xs_pad, ilens)
#
#         xs_pad = self.dropout_cnn(xs_pad)
#         xs_pad = self.cnn_proj(xs_pad)
#
#         self.h_length = xs_pad.size(1)
#         batch = len(ilens)
#
#         self.mask = to_device(self, make_pad_mask(ilens))
#
#         gt_01 = torch.tanh(self.global_attention_mlp_01(xs_pad))
#         gt_02 = self.global_attention_mlp_02(gt_01).squeeze(2)
#         # NOTE consider zero padding when compute gt_02.
#         gt_02.masked_fill_(self.mask, -float('inf'))
#         gw0 = F.softmax(self.gatt_scale * gt_02, dim=1)
#         ga0 = self.g_adapt_0 + torch.sum(xs_pad * gw0.view(batch, self.h_length, 1), dim=1)
#
#         # xs_pad = ga0.view(batch, 1, -1) + xs_pad
#         xs_pad = torch.cat((xs_pad, ga0.view(batch, 1, -1)), dim=1)
#         ilens = [val + 1 for val in ilens]
#         self.mask = to_device(self, make_pad_mask(ilens))
#
#         self.h_length = self.h_length + 1
#
#         self.layer_1.flatten_parameters()
#         self.layer_2.flatten_parameters()
#         self.layer_3.flatten_parameters()
#
#         xs_pack = pack_padded_sequence(xs_pad, ilens, batch_first=True)
#
#         # first BLSTM Layer
#         h1, _ = self.layer_1(xs_pack)
#         h1, ilens = pad_packed_sequence(h1, batch_first=True)
#         h1 = self.dropout_1(h1)  # Take the global attention over this.
#
#         gt_11 = torch.tanh(self.global_attention_mlp_11(h1))
#         gt_12 = self.global_attention_mlp_12(gt_11).squeeze(2)
#         # NOTE consider zero padding when compute gt_12.
#         gt_12.masked_fill_(self.mask, -float('inf'))
#         gw1 = F.softmax(self.gatt_scale * gt_12, dim=1)
#         ga1 = self.g_adapt_1 + torch.sum(h1 * gw1.view(batch, self.h_length, 1), dim=1)
#
#         # h1 = ga1.view(batch, 1, -1) + h1
#         h1 = torch.cat((h1, ga1.view(batch, 1, -1)), dim=1)
#
#         xs_pack = pack_padded_sequence(h1, ilens, batch_first=True)
#
#         # Second BLSTM Layer
#         h2, _ = self.layer_2(xs_pack)
#         h2, ilens = pad_packed_sequence(h2, batch_first=True)
#         h2 = self.dropout_2(h2)  # Take the global attention over this.
#
#         gt_21 = torch.tanh(self.global_attention_mlp_21(h2))
#         gt_22 = self.global_attention_mlp_22(gt_21).squeeze(2)
#         # NOTE consider zero padding when compute gt_12.
#         gt_22.masked_fill_(self.mask, -float('inf'))
#         gw2 = F.softmax(self.gatt_scale * gt_22, dim=1)
#         ga2 = self.g_adapt_2 + torch.sum(h2 * gw2.view(batch, self.h_length, 1), dim=1)
#
#         #         h2 = ga2.view(batch, 1, -1) + h2
#         h2 = torch.cat((h2, ga2.view(batch, 1, -1)), dim=1)
#
#         xs_pack = pack_padded_sequence(h2, ilens, batch_first=True)
#
#         # Third BLSTM Layer
#         h3, _ = self.layer_3(xs_pack)
#         h3, ilens = pad_packed_sequence(h3, batch_first=True)
#         #         h3 = self.dropout_3(h3)   #Take the global attention over this.
#
#         gt_31 = torch.tanh(self.global_attention_mlp_31(h3))
#         gt_32 = self.global_attention_mlp_32(gt_31).squeeze(2)
#         # NOTE consider zero padding when compute gt_12.
#         gt_32.masked_fill_(self.mask, -float('inf'))
#         gw3 = F.softmax(self.gatt_scale * gt_32, dim=1)
#         ga3 = self.g_adapt_3 + torch.sum(h3 * gw3.view(batch, self.h_length, 1), dim=1)
#
#         #         h3 = ga3.view(batch, 1, -1) + h3
#         h3 = torch.cat((h3, ga3.view(batch, 1, -1)), dim=1)
#
#         global_attention = self.weight_0 * ga0 + self.weight_1 * ga1 + self.weight_2 * ga2 + self.weight_3 * ga3
#         #         global_attention = self.weight_1 * ga1 + self.weight_2 * ga2 + self.weight_3 * ga3 + self.weight_4 * ga4
#         ga_projected = torch.tanh(self.l_gt_last(
#             global_attention.contiguous()))
#         # ga_projected = ga_projected.view(global_attention.size(0), global_attention.size(1), -1)
#
#         projected = torch.tanh(self.l_last(
#             h3.contiguous().view(-1, h3.size(2))))
#         projected = projected.view(h3.size(0), h3.size(1), -1)
#
#         return projected, ilens, ga_projected

# class MultiLayerGlobalAttentionEncoder(torch.nn.Module):
#     def __init__(self, in_dim, hid, in_channel=1):
#         super(MultiLayerGlobalAttentionEncoder, self).__init__()
#
#         self.vgg = VGG2L(in_channel)
#         self.dropout_cnn = torch.nn.Dropout(0.2)
#
#         self.gatt_scale = 0.7
#         self.h_length = None
#
#         self.layer_1 = torch.nn.LSTM(get_vgg2l_odim(in_dim, in_channel=in_channel), hid, 1, batch_first=True,
#                                      bidirectional=True)
#         self.layer_2 = torch.nn.LSTM(2 * hid, hid, 1, batch_first=True, bidirectional=True)
#         self.layer_3 = torch.nn.LSTM(2 * hid, hid, 1, batch_first=True, bidirectional=True)
#
#         self.l_last = torch.nn.Linear(2 * hid, hid)
#         self.l_gt_last = torch.nn.Linear(2 * hid, hid)
#
#         self.dropout_1 = torch.nn.Dropout(0.2)
#         self.dropout_2 = torch.nn.Dropout(0.2)
#
#         self.global_attention_mlp_11 = torch.nn.Linear(2 * hid, hid, bias=False)
#         self.global_attention_mlp_12 = torch.nn.Linear(hid, 1, bias=False)
#
#         self.global_attention_mlp_21 = torch.nn.Linear(2 * hid, hid, bias=False)
#         self.global_attention_mlp_22 = torch.nn.Linear(hid, 1, bias=False)
#
#         self.global_attention_mlp_31 = torch.nn.Linear(2 * hid, hid, bias=False)
#         self.global_attention_mlp_32 = torch.nn.Linear(hid, 1, bias=False)
#
#         self.g_adapt_1 = torch.nn.Parameter(torch.zeros(1, 2 * hid, requires_grad=False))
#         self.g_adapt_2 = torch.nn.Parameter(torch.zeros(1, 2 * hid, requires_grad=False))
#         self.g_adapt_3 = torch.nn.Parameter(torch.zeros(1, 2 * hid, requires_grad=False))
#
#         self.weight_1 = torch.nn.Parameter(torch.Tensor([0.333]))
#         self.weight_2 = torch.nn.Parameter(torch.Tensor([0.333]))
#         self.weight_3 = torch.nn.Parameter(torch.Tensor([0.333]))
#
#     def forward(self, xs_pad, ilens):
#         logging.info(self.__class__.__name__ + ' Forward is called ')
#
#         xs_pad, ilens, states = self.vgg(xs_pad, ilens)
#         xs_pad = self.dropout_cnn(xs_pad)
#
#         self.h_length = xs_pad.size(1)
#         batch = len(ilens)
#         self.mask = to_device(self, make_pad_mask(ilens))
#
#         self.layer_1.flatten_parameters()
#         self.layer_2.flatten_parameters()
#         self.layer_3.flatten_parameters()
#         #         self.layer_4.flatten_parameters()
#
#         xs_pack = pack_padded_sequence(xs_pad, ilens, batch_first=True)
#
#         # first BLSTM Layer
#         h1, _ = self.layer_1(xs_pack)
#         h1, ilens = pad_packed_sequence(h1, batch_first=True)
#         h1 = self.dropout_1(h1)  # Take the global attention over this.
#
#         gt_11 = torch.tanh(self.global_attention_mlp_11(h1))
#         gt_12 = self.global_attention_mlp_12(gt_11).squeeze(2)
#         # NOTE consider zero padding when compute gt_12.
#         gt_12.masked_fill_(self.mask, -float('inf'))
#         gw1 = F.softmax(self.gatt_scale * gt_12, dim=1)
#         ga1 = self.g_adapt_1 + torch.sum(h1 * gw1.view(batch, self.h_length, 1), dim=1)
#
#         xs_pack = pack_padded_sequence(h1, ilens, batch_first=True)
#
#         # Second BLSTM Layer
#         h2, _ = self.layer_2(xs_pack)
#         h2, ilens = pad_packed_sequence(h2, batch_first=True)
#         h2 = self.dropout_2(h2)  # Take the global attention over this.
#
#         gt_21 = torch.tanh(self.global_attention_mlp_21(h2))
#         gt_22 = self.global_attention_mlp_22(gt_21).squeeze(2)
#         # NOTE consider zero padding when compute gt_12.
#         gt_22.masked_fill_(self.mask, -float('inf'))
#         gw2 = F.softmax(self.gatt_scale * gt_22, dim=1)
#         ga2 = self.g_adapt_2 + torch.sum(h2 * gw2.view(batch, self.h_length, 1), dim=1)
#
#         xs_pack = pack_padded_sequence(h2, ilens, batch_first=True)
#
#         # Third BLSTM Layer
#         h3, _ = self.layer_3(xs_pack)
#         h3, ilens = pad_packed_sequence(h3, batch_first=True)
#         #         h3 = self.dropout_3(h3)   #Take the global attention over this.
#
#         gt_31 = torch.tanh(self.global_attention_mlp_31(h3))
#         gt_32 = self.global_attention_mlp_32(gt_31).squeeze(2)
#         # NOTE consider zero padding when compute gt_12.
#         gt_32.masked_fill_(self.mask, -float('inf'))
#         gw3 = F.softmax(self.gatt_scale * gt_32, dim=1)
#         ga3 = self.g_adapt_3 + torch.sum(h3 * gw3.view(batch, self.h_length, 1), dim=1)
#
#         global_attention = self.weight_1 * ga1 + self.weight_2 * ga2 + self.weight_3 * ga3
#         ga_projected = torch.tanh(self.l_gt_last(
#             global_attention.contiguous()))
#
#         projected = torch.tanh(self.l_last(
#             h3.contiguous().view(-1, h3.size(2))))
#         projected = projected.view(h3.size(0), h3.size(1), -1)
#
#         return projected, ilens, ga_projected


def encoder_for(args, idim, subsample):
    if args.gunits:
        return MultiLayerGlobalAttentionEncoder(idim, args.eunits)
    else:
        return Encoder(args.etype, idim, args.elayers, args.eunits, args.eprojs, subsample, args.dropout_rate)
