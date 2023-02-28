#! /usr/bin/env python3
#

"""The dimension related operations."""

import itertools

import torch
import torch.nn as nn

from .sparse_tensor import SparseTensor
from jactorch.functional import broadcast
from ._utils import sparse_exclude_mask, sparse_mask_value

__all__ = ['SparseExpander', 'SparseReducer', 'SparsePermutation']


class SparseExpander(nn.Module):
    """Capture a free variable into predicates, implemented by broadcast."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, inputs):
        for b_id, t in enumerate(inputs):
            inputs[b_id] = t.expand()
        return inputs

    def get_output_dim(self, input_dim):
        return input_dim


class SparseReducer(nn.Module):
    """Reduce out a variable via quantifiers (exists/forall), implemented by max/min-pooling."""

    def __init__(self, dim, exclude_self=True, max=True, min=True, mean=True, sum=True):
        super().__init__()
        self.dim = dim
        self.exclude_self = exclude_self
        self.max = max
        self.min = min
        self.mean = mean
        self.sum = sum

    def forward(self, inputs):
        """[summary]

        Args:
            inputs (List[SparseTensor]): a list of sparse tensor

        Returns:
            List[SparseTensor]: a list of sparse tensor 
        """
        outputs = []
        for v in inputs:
            if self.dim > 1:
                exclude_mask = v.exclude_mask()
                v.val[exclude_mask, :] = 0
                outputs.append(v.reduce_arity('max'))
            else:
                outputs.append(v.reduce_arity('max'))
        return outputs

    def get_output_dim(self, input_dim):
        return input_dim


class SparsePermutation(nn.Module):
    """Create r! new predicates by permuting the axies for r-arity predicates."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, inputs):
        if self.dim <= 1:
            return inputs
        res = []
        for b_id, t in enumerate(inputs):
            res.append(t.all_permutation())
        return res

    def get_output_dim(self, input_dim):
        mul = 1
        for i in range(self.dim):
            mul *= i + 1
        return input_dim * mul
