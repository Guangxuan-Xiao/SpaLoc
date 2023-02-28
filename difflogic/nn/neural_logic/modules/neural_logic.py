#! /usr/bin/env python3
#

"""MLP-based implementation for logic and logits inference."""

import torch.nn as nn
import torch
from jactorch.quickstart.models import MLPModel
from .sparse_tensor import SparseTensor
from sparse_hypergraph import SparseHypergraph
__all__ = ['LogicInference', 'LogitsInference']


class InferenceBase(nn.Module):
    """MLP model with shared parameters among other axies except the channel axis."""

    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.layer = nn.Sequential(
            MLPModel(input_dim, output_dim, hidden_dim, flatten=False))

    def forward(self, inputs):
        # print(inputs.shape)
        input_size = inputs.size()[:-1]
        input_channel = inputs.size(-1)

        f = inputs.view(-1, input_channel)
        f = self.layer(f)
        f = f.view(*input_size, -1)
        return f

    def get_output_dim(self, input_dim):
        return self.output_dim


class SparseInferenceBase(nn.Module):
    """MLP model with shared parameters among other axies except the channel axis."""

    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.layer = nn.Sequential(
            MLPModel(input_dim, output_dim, hidden_dim, flatten=False))

    def forward(self, inputs):
        val = torch.cat([t.val for t in inputs], dim=0)
        val = self.layer(val)
        current_idx = 0
        outputs = []
        for b_id, t in enumerate(inputs):
            next_idx = current_idx + t.num_entries()
            outputs.append(SparseHypergraph(
                val[current_idx:next_idx], t.n, t.arity, self.output_dim, t.coo))
            current_idx = next_idx
        return outputs

    def get_output_dim(self, input_dim):
        return self.output_dim


class LogicInference(InferenceBase):
    """MLP layer with sigmoid activation."""

    def __init__(self, input_dim, output_dim, hidden_dim, norm='sigmoid'):
        super().__init__(input_dim, output_dim, hidden_dim)
        if norm == 'sigmoid':
            act = nn.Sigmoid()
        elif norm == 'tanh':
            act = nn.Tanh()
        elif norm == 'relu':
            act = nn.ReLU()
        elif norm is None:
            act = None
        else:
            raise NotImplementedError
        if act is not None:
            self.layer.add_module(str(len(self.layer)), act)


class SparseLogicInference(SparseInferenceBase):
    """MLP layer with sigmoid activation."""

    def __init__(self, input_dim, output_dim, hidden_dim, norm='sigmoid'):
        super().__init__(input_dim, output_dim, hidden_dim)
        if norm == 'sigmoid':
            act = nn.Sigmoid()
        elif norm == 'tanh':
            act = nn.Tanh()
        elif norm == 'relu':
            act = nn.ReLU()
        else:
            raise NotImplementedError
        self.layer.add_module(str(len(self.layer)), act)


class LogitsInference(InferenceBase):
    pass
