#! /usr/bin/env python3
#

"""The dimension related operations."""

import itertools

import torch
import torch.nn as nn

from jactorch.functional import broadcast

from ._utils import exclude_mask, mask_value, print_groundings

__all__ = ['Expander', 'Reducer', 'Permutation']


class Expander(nn.Module):
    """Capture a free variable into predicates, implemented by broadcast."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, inputs, n=None):
        if self.dim == 0:
            assert n is not None
        elif n is None:
            n = inputs.size(self.dim)
        dim = self.dim + 1
        return broadcast(inputs.unsqueeze(dim), dim, n)

    def get_output_dim(self, input_dim):
        return input_dim


class Reducer(nn.Module):
    """Reduce out a variable via quantifiers (exists/forall), implemented by max/min-pooling."""

    def __init__(self, dim, exclude_self=True, forall=True, mean=False):
        super().__init__()
        self.dim = dim
        self.exclude_self = exclude_self
        self.forall = forall
        self.mean = mean

    def forward(self, inputs):
        shape = inputs.size()
        inp0, inp1 = inputs, inputs
        if self.exclude_self:
            mask = exclude_mask(inputs, cnt=self.dim, dim=-1 - self.dim)
            inp0 = mask_value(inputs, mask, 0.0)
            inp1 = mask_value(inputs, mask, 1.0)
        output = []
        exists = torch.max(inp0, dim=-2)[0]
        output.append(exists)
        dim = 1

        if self.mean:
            mean = torch.mean(inp0, dim=-2)
            output.append(mean)
            dim += 1

        if self.forall:
            forall = torch.min(inp1, dim=-2)[0]
            output.append(forall)
            dim += 1
        shape = shape[:-2] + (shape[-1] * dim,)
        return torch.stack(output, dim=-1).view(shape)

    def get_output_dim(self, input_dim):
        output_dim = input_dim
        if self.mean:
            output_dim += input_dim
        if self.forall:
            output_dim += input_dim
        return output_dim


class Permutation(nn.Module):
    """Create r! new predicates by permuting the axies for r-arity predicates."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, inputs):
        if self.dim <= 1:
            return inputs
        nr_dims = len(inputs.size())
        # Assume the last dim is channel.
        index = tuple(range(nr_dims - 1))
        start_dim = nr_dims - 1 - self.dim
        assert start_dim > 0
        res = []
        for i in itertools.permutations(index[start_dim:]):
            p = index[:start_dim] + i + (nr_dims - 1,)
            res.append(inputs.permute(p))
        return torch.cat(res, dim=-1)

    def get_output_dim(self, input_dim):
        mul = 1
        for i in range(self.dim):
            mul *= i + 1
        return input_dim * mul

