import math
import pdb

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import Tensor
from functools import partial
from typing import Optional

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


DOM_TYPE_ALL = ('spatial', 'temporal')

def _assert_dom_type(dom_type: str) :
    assert dom_type in DOM_TYPE_ALL, \
        f"Domain type must be one of {DOM_TYPE_ALL}, but got {dom_type}"

class PositionalEncoding(nn.Module) :
    def __init__(self,
        n_joints: int,
        seq_len: int,
        d_model: int,
        dom_type: str,
    ) -> None:

        super().__init__()

        _assert_dom_type(dom_type)

        self.n_joints = n_joints
        self.seq_len = seq_len
        self.dom_type = dom_type

        dtype = torch.get_default_dtype()

        if dom_type == 'temporal':
            pos_list = list(range(self.n_joints * self.seq_len))

        elif dom_type == 'spatial' :
            pos_list = []
            for t in range(self.seq_len) :
                for j_id in range(self.n_joints) :
                    pos_list.append(j_id)

        position = torch.from_numpy(np.array(pos_list)).unsqueeze(1).to(dtype)

        pe = torch.zeros(self.seq_len * self.n_joints, d_model)

        div_term = torch.exp(torch.arange(0, d_model, 2).to(dtype) *
                             ( -math.log(10000.0) / d_model ))
        pe[:, 0::2] = torch.sin(position* div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe.unsqueeze_(0)
        self.register_buffer('pe', pe)


    def forward(self, x: Tensor ) -> Tensor :
        """
        Args:
        x: Tensor, shape [batch_size, seq_len * n_joints, d_model]
        """
        x = x + self.pe[:, :x.size(1)]

        return x

class LayerNorm(nn.Module):
    """Construct a layernorm module (See citation for details)."""
    def __init__(self,
        d_model: int,
        eps: float = 1e-6,
    ) -> None :

        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(d_model))
        self.b_2 = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor :
        """
        Args:
        x: Tensor, shape [batch_size, seq_len, d_model]
        """
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        x = self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        return x

class QkvProjector(nn.Module) :
    """Project Q, K, V for onto desired attention blocks."""
    def __init__(self,
        d_in: int,
        n_heads: int,
        d_head: int,
    ) -> None :

        super().__init__()

        self.n_heads = n_heads
        self.d_head = d_head

        self.projector = nn.Linear(d_in, self.n_heads * self.d_head)

    def forward(self, x: Tensor) -> Tensor :
        bs = x.size(0)
        x = self.projector(x) # bs x n x n_h * d_h
        x = x.view(bs, -1, self.n_heads, self.d_head) # bs x n x n_h x d_h
        x = x.transpose(1, 2) # bs x n_h x n x d_h

        return x

class MultiHeadedAttention(nn.Module) :
    def __init__(self,
        n_heads: int,
        d_head: int,
        d_in: int,
        dom_type: str,
        seq_len: int,
        n_joints: int,
        dropout: float = 0.,
        eps: float = 1e-6,
    ) -> None :

        super().__init__()

        _assert_dom_type(dom_type)

        self.d_head = d_head
        self.n_heads = n_heads
        self.dom_type = dom_type
        self.seq_len = seq_len
        self.n_joints = n_joints

        self.d_model = self.n_heads * self.d_head
        self.eps = eps

        st_mask = self.get_spatiotemporal_mask()
        self.register_buffer('scaled_st_mask', st_mask / math.sqrt(self.d_head))
        self.register_buffer('th_logit',
                (1 - st_mask) * torch.finfo(torch.get_default_dtype()).min)

        if dropout < 1e-3 :
            self.k_map = QkvProjector(d_in, self.n_heads, self.d_head)
            self.q_map = QkvProjector(d_in, self.n_heads, self.d_head)
            self.v_map = nn.Sequential(
                            QkvProjector(d_in, self.n_heads, self.d_head),
                            nn.ReLU(),
            )

        else :
            self.k_map = nn.Sequential(
                            QkvProjector(d_in, self.n_heads, self.d_head),
                            nn.Dropout(dropout),
            )

            self.q_map = nn.Sequential(
                            QkvProjector(d_in, self.n_heads, self.d_head),
                            nn.Dropout(dropout),
            )

            self.v_map = nn.Sequential(
                            QkvProjector(d_in, self.n_heads, self.d_head),
                            nn.ReLU(),
                            nn.Dropout(dropout),
            )

    def get_spatiotemporal_mask(self) :
        n_pts = self.seq_len * self.n_joints
        s_mask = torch.zeros(n_pts, n_pts)

        for i in range(self.seq_len) :
            i_begin = i * self.n_joints
            i_end = i_begin + self.n_joints
            s_mask[i_begin:i_end, i_begin:i_end].fill_(1) 

        if self.dom_type == 'spatial' :
            return s_mask

        t_mask = 1 - s_mask + torch.eye(n_pts)
        return t_mask

    def compute_qkv_attention(self,
        q: Tensor, 
        k: Tensor,
        v: Tensor,
        adj: Optional[Tensor] = None,
    ) -> Tensor :

        scores = q @ k.transpose(-2, -1)
        scores = (scores * self.scaled_st_mask) + self.th_logit
        if isinstance(adj, np.ndarray):
            adj = torch.from_numpy(adj).float()
        if adj is not None:
            if adj.dim() == 2:
                adj = adj.unsqueeze(0).expand(q.size(0), -1, -1)
            scores = scores * adj.unsqueeze(1)
        tmp_max, _ = scores.max(dim=-1, keepdim=True)
        scores = torch.exp(scores - tmp_max)
        scores = scores / (scores.sum(dim=-1, keepdim=True) + self.eps)

        out = scores @ v
        return out


    def forward(self, x, adj: Optional[Tensor] = None):
        """
        Args :
        x : Tensor, shape [bs x n x n_h * d_h]
        adj : Optional[Tensor], shape [bs x n x n]
        Returns :
        x : Tensor, shape [bs x n x n_h * d_h]
        """
        bs = x.size(0) # bs x n x n_h * d_h
        q = self.q_map(x) # bs x n_h x n x d_h
        k = self.k_map(x) # bs x n_h x n x d_h
        v = self.v_map(x) # bs x n_h x n x d_h
        x = self.compute_qkv_attention(q, k, v, adj) # bs x n_h x n x d_h
        x = x.transpose(1, 2).contiguous() # bs x n x n_h x d_h
        x = x.view(bs, -1, self.d_model) # bs x n x n_h * d_h

        return x

class SpatioTemporalAttention(nn.Module):
    """Encoder is made up of self-attn and feed forward (defined below)"""

    def __init__(self,
                 d_in: int,
                 d_out: int,
                 n_heads: int,
                 d_head: int,
                 seq_len: int,
                 n_joints: int,
                 dom_type: str,
                 dropout: float = 0.,
                 ) -> None :

        super().__init__()
        self.pe = PositionalEncoding(n_joints, seq_len, d_in, dom_type)

        self.att_layer = MultiHeadedAttention(
                        n_heads,
                        d_head,
                        d_in,
                        dom_type,
                        seq_len,
                        n_joints,
                        dropout,
        )

        layers = [
            nn.Linear(n_heads * d_head, d_out),
            nn.ReLU(),
            LayerNorm(d_out),
        ]

        if dropout < 1e-3 :
            layers.append(nn.Dropout(dropout))

        self.linear = nn.Sequential(*layers)

        self.init_parameters()

    def forward(self,
        x: Tensor,
        adj: Optional[Tensor]
    ) -> Tensor :
        x = self.pe(x) #add PE
        x = self.att_layer(x, adj)
        x = self.linear(x)
        return x

    def init_parameters( self ) :
        model_list = [ self.att_layer, self.linear]
        for model in model_list :
            for p in model.parameters() :
                if p.dim() > 1 :
                    nn.init.xavier_uniform_(p)

def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)

def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)


class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size-1) * (dilation-1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class MultiScale_TemporalConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilations=[1,2,3,4],
                 residual=True,
                 residual_kernel_size=1):

        super().__init__()
        assert out_channels % (len(dilations) + 2) == 0, '# out channels should be multiples of # branches'

        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches
        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size]*len(dilations)
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    padding=0),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                TemporalConv(
                    branch_channels,
                    branch_channels,
                    kernel_size=ks,
                    stride=stride,
                    dilation=dilation),
            )
            for ks, dilation in zip(kernel_size, dilations)
        ])

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,1), stride=(stride,1), padding=(1,0)),
            nn.BatchNorm2d(branch_channels)
        ))

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride,1)),
            nn.BatchNorm2d(branch_channels)
        ))

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)

        self.apply(weights_init)

    def forward(self, x):
        # Input dim: (N,C,T,V)
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)
        out = torch.cat(branch_outs, dim=1)
        out += res
        return out

class TDGC(nn.Module):
    def __init__(self, in_channels, out_channels, rel_reduction=8, mid_reduction=1):
        super(TDGC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels == 3 or in_channels == 9:
            self.rel_channels = 8
            self.mid_channels = 16
        else:
            self.rel_channels = in_channels // rel_reduction
            self.mid_channels = in_channels // mid_reduction
        self.conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)

        self.tanh = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, x, A=None, alpha=1, beta=1, gamma=0.1):
        x1, x3 = self.conv1(x).mean(-2), self.conv3(x)
        x1 = self.tanh(x1.unsqueeze(-1) - x1.unsqueeze(-2))
        x1 = self.conv4(x1) * alpha + (A.unsqueeze(0).unsqueeze(0) if A is not None else 0)
        A_S = x1
        x1 = torch.einsum('ncuv,nctv->nctu', x1, x3)
        x4 = self.tanh(x3.mean(-3).unsqueeze(-1) - x3.mean(-3).unsqueeze(-2))
        A_T = x4
        x3 = x3.permute(0, 2, 1, 3)
        x5 = torch.einsum('btmn,btcn->bctm', x4, x3)
        x1 = x1 * beta  + x5 * gamma
        return x1, A_S, A_T

class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, adaptive=True, residual=True):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels
        self.adaptive = adaptive
        self.num_subset = A.shape[0]
        self.convs = nn.ModuleList()
        for i in range(self.num_subset):
            self.convs.append(TDGC(in_channels, out_channels))

        if residual:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.down = lambda x: x
        else:
            self.down = lambda x: 0
        if self.adaptive:
            self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        else:
            self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.alpha = nn.Parameter(torch.zeros(1)) # No I
        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU(inplace=True)

        self.beta = nn.Parameter(torch.tensor(0.5)) # 1.0 1.4 2.0
        self.gamma = nn.Parameter(torch.tensor(0.1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

    def forward(self, x):
        y = None
        A_Si_list = []
        A_Ti_list = []
        if self.adaptive:
            A = self.PA
        else:
            A = self.A.cuda(x.get_device())
        for i in range(self.num_subset):
            z, A_Si, A_Ti = self.convs[i](x, A[i], self.alpha, self.beta, self.gamma)
            A_Si_list.append(A_Si)
            A_Ti_list.append(A_Ti)
            y = z + y if y is not None else z
        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)
        A_S_tensor = torch.stack(A_Si_list, dim=1).mean(dim=1)
        A_T_tensor = torch.stack(A_Ti_list, dim=1).mean(dim=1)
        return y, A_S_tensor,  A_T_tensor


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, adaptive=True, kernel_size=5, dilations=[1,2]):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A, adaptive=adaptive)
        self.tcn1 = MultiScale_TemporalConv(out_channels, out_channels, kernel_size=kernel_size, stride=stride, dilations=dilations,
                                            residual=False)
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        fea, A_s, A_t = self.gcn1(x)
        y = self.relu(self.tcn1(fea) + self.residual(x))
        return y, A_s, A_t

class Model(nn.Module):
    def __init__(self,
                 n_classes=14,
                 in_channels=3,
                 n_heads=8,
                 d_head=16,
                 d_feat=128,
                 seq_len=60,
                 n_joints=22,
                 dropout=0,
                 num_person=1,
                 graph=None,
                 graph_args=dict(),
                 drop_out=0,
                 adaptive=True,
                 ):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)
        A = self.graph.A

        self.data_bn = nn.BatchNorm1d(num_person * in_channels * n_joints)

        base_channel = 64
        self.l1 = TCN_GCN_unit(in_channels, base_channel, A, residual=False, adaptive=adaptive)
        self.l2 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l3 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l4 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)

        st_layer  = partial(SpatioTemporalAttention,
                           d_in=d_feat,
                           d_out=d_feat,
                           n_heads=n_heads,
                           d_head=d_head,
                           seq_len=seq_len,
                           n_joints=n_joints,
                           dropout=dropout)

        self.spatial_att = st_layer(dom_type='spatial')
        self.temporal_att = st_layer(dom_type='temporal')

        self.feat_linear = nn.Linear(base_channel, d_feat)
        self.inv_feat_linear = nn.Linear(d_feat, base_channel)

        self.l5 = TCN_GCN_unit(base_channel, base_channel*2, A, stride=2, adaptive=adaptive)
        self.l6 = TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l7 = TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l8 = TCN_GCN_unit(base_channel*2, base_channel*4, A, stride=2, adaptive=adaptive)
        self.l9 = TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)
        self.l10 = TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)

        self.final = nn.Linear(base_channel * 4, n_classes)
        nn.init.normal_(self.final.weight, 0, math.sqrt(2. / n_classes))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x


    def forward_feature(self, x):
        x = x.permute(0, 3, 1, 2).contiguous().unsqueeze(-1);
        if len(x.shape) == 3:
            B, T, VC = x.shape
            x = x.view(B, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        B, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(B, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(B, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(B * M, C, T, V)
        x, _, _ = self.l1(x)
        x, _, _ = self.l2(x)
        x, _, _ = self.l3(x)
        x, AS, AT= self.l4(x)
        N = T * V
        At = AT.new_zeros((B, N, N))
        for b in range(B):
            for t in range(T):
                i0 = t * V
                At[b, i0:i0 + V, i0:i0 + V] = AT[b, t]
        AS_mean = AS.mean(dim=1)
        As = AT.new_zeros((B, N, N))
        for b in range(B):
            for t in range(T):
                i0 = t * V
                As[b, i0:i0 + V, i0:i0 + V] = AS_mean[b]
        _, C_sta ,_ ,_ = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(B, T * V, C_sta)
        x = self.feat_linear(x)
        x = self.spatial_att(x, As)
        x = self.temporal_att(x, At)
        x = self.inv_feat_linear(x)
        x = x.view(B, T, V, -1).permute(0, 3, 1, 2).contiguous()
        x, _, _ = self.l5(x)
        x, _, _ = self.l6(x)
        x, _, _ = self.l7(x)
        x, _, _ = self.l8(x)
        x, _, _ = self.l9(x)
        x, _, _ = self.l10(x)
        _, C_new, _, _ = x.shape
        x = x.view(B, M, C_new, -1)
        x = x.mean(3).mean(1)
        x = self.drop_out(x)
        return x


    def forward(self, x):
        # input x: [batch_size, seq_len, n_joints, in_channels]
        x = self.forward_feature(x);
        x = self.final(x)
        return x


    def forward_frame_features(self, x):
        """
        Returns per-frame features without final spatiotemporal global average
        Input:
            x: [B, T, V, C]
        Output:
            frame_feats: [B, T, D]
                T = seq_len
                D = channel dimension (equal to channel count after inv_feat_linear)
        """
        x = x.permute(0, 3, 1, 2).contiguous().unsqueeze(-1);
        if len(x.shape) == 3:
            B, T, VC = x.shape
            x = x.view(B, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        B, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(B, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(B, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(B * M, C, T, V)
        x, _, _ = self.l1(x)
        x, _, _ = self.l2(x)
        x, _, _ = self.l3(x)
        x, AS, AT= self.l4(x)
        N = T * V
        At = AT.new_zeros((B, N, N))
        for b in range(B):
            for t in range(T):
                i0 = t * V
                At[b, i0:i0 + V, i0:i0 + V] = AT[b, t]
        AS_mean = AS.mean(dim=1)
        As = AT.new_zeros((B, N, N))
        for b in range(B):
            for t in range(T):
                i0 = t * V
                As[b, i0:i0 + V, i0:i0 + V] = AS_mean[b]
        _, C_sta ,_ ,_ = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(B, T * V, C_sta)
        x = self.feat_linear(x)
        x = self.spatial_att(x, As)
        x = self.temporal_att(x, At)
        x = self.inv_feat_linear(x)
        x = x.view(B, T, V, -1)
        frame_feats = x.mean(dim=2)

        return frame_feats
