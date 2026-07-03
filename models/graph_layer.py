import torch
from torch.nn import Parameter, Linear
import torch.nn as nn
import torch.nn.functional as F


def glorot(tensor):
    if tensor is not None:
        nn.init.xavier_uniform_(tensor)


def zeros(tensor):
    if tensor is not None:
        nn.init.zeros_(tensor)


def remove_self_loops(edge_index):
    if edge_index.numel() == 0:
        return edge_index
    mask = edge_index[0] != edge_index[1]
    return edge_index[:, mask]


def add_self_loops(edge_index, num_nodes, device):
    loop_index = torch.arange(num_nodes, device=device, dtype=torch.long).unsqueeze(0).repeat(2, 1)
    return torch.cat([edge_index, loop_index], dim=1)


def grouped_softmax(alpha, target_index, num_nodes):
    index = target_index.view(-1, 1, 1).expand_as(alpha)
    max_per_node = torch.full(
        (num_nodes, alpha.size(1), alpha.size(2)),
        -torch.inf,
        dtype=alpha.dtype,
        device=alpha.device,
    )
    max_per_node.scatter_reduce_(0, index, alpha, reduce='amax', include_self=True)
    exp_alpha = torch.exp(alpha - max_per_node[target_index])
    denom = torch.zeros_like(max_per_node)
    denom.scatter_add_(0, index, exp_alpha)
    return exp_alpha / (denom[target_index] + 1e-16)


class GraphLayer(nn.Module):
    def __init__(self, in_channels, out_channels, heads=1, concat=True,
                 negative_slope=0.2, dropout=0, bias=True, inter_dim=-1, **kwargs):
        super(GraphLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.__alpha__ = None

        self.lin = Linear(in_channels, heads * out_channels, bias=False)

        self.att_i = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_j = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_em_i = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_em_j = Parameter(torch.Tensor(1, heads, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin.weight)
        glorot(self.att_i)
        glorot(self.att_j)
        zeros(self.att_em_i)
        zeros(self.att_em_j)
        zeros(self.bias)

    def forward(self, x, edge_index, embedding, return_attention_weights=False):
        if torch.is_tensor(x):
            x = self.lin(x)
        else:
            x = self.lin(x[0])

        num_nodes = x.size(0)
        edge_index = remove_self_loops(edge_index.long())
        edge_index = add_self_loops(edge_index, num_nodes, x.device)

        source = edge_index[0]
        target = edge_index[1]

        x_i = x[target].view(-1, self.heads, self.out_channels)
        x_j = x[source].view(-1, self.heads, self.out_channels)

        if embedding is not None:
            embedding_i = embedding[target].unsqueeze(1).repeat(1, self.heads, 1)
            embedding_j = embedding[source].unsqueeze(1).repeat(1, self.heads, 1)
            key_i = torch.cat((x_i, embedding_i), dim=-1)
            key_j = torch.cat((x_j, embedding_j), dim=-1)
        else:
            key_i = x_i
            key_j = x_j

        cat_att_i = torch.cat((self.att_i, self.att_em_i), dim=-1)
        cat_att_j = torch.cat((self.att_j, self.att_em_j), dim=-1)
        alpha = (key_i * cat_att_i).sum(-1) + (key_j * cat_att_j).sum(-1)
        alpha = F.leaky_relu(alpha.unsqueeze(-1), self.negative_slope)
        alpha = grouped_softmax(alpha, target, num_nodes)

        if return_attention_weights:
            self.__alpha__ = alpha

        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        messages = x_j * alpha

        out = torch.zeros(num_nodes, self.heads, self.out_channels, device=x.device, dtype=x.dtype)
        out.index_add_(0, target, messages)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        if return_attention_weights:
            alpha, self.__alpha__ = self.__alpha__, None
            return out, (edge_index, alpha)
        return out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.heads
        )
