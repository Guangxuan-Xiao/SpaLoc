import torch
import numpy as np
from torch_scatter import scatter, segment_coo
from copy import copy
import itertools
from torch.linalg import norm
import functools
# TODO: Add argument to control this
EPSILON = 1e-2


class SparseTensor:
    def __init__(self, val, n, arity, channel, coo=None):
        self.n = n
        self.arity = arity
        self.channel = channel
        self.shape = tuple([self.n for _ in range(self.arity)])
        self.space_size = n ** arity
        self.device = val.device
        self.dtype = val.dtype
        self.val = val.view(-1, self.channel)
        if coo is None:
            self.coo = self._default_coo()
        else:
            self.coo = coo

    def sparsify_(self, eps=EPSILON):
        if self.length == 0:
            return
        mask = self.val.abs().max(-1)[0] > eps
        self.val = self.val[mask]
        self.coo = self.coo[mask]

    def sparsify_by_(self, t, eps=EPSILON):
        if self.length == 0:
            return
        mask = t.val.abs().max(-1)[0] > eps
        self.val = self.val[mask]
        self.coo = self.coo[mask]

    def _default_coo(self):
        return torch.arange(self.space_size, device=self.device, dtype=torch.long, requires_grad=False)

    def to_dense(self):
        if self.length == self.space_size:
            return self.val.view(*self.shape, self.channel)
        else:
            dense = torch.zeros((self.space_size, self.channel),
                                dtype=self.dtype, device=self.device)
            dense[self.coo] = self.val
            return dense.view(*self.shape, self.channel)

    @property
    def nnz(self):
        if self.length == 0:
            return 0
        return (self.val.abs().max(-1)[0] > EPSILON).sum()

    @property
    def length(self):
        return len(self.coo)

    def sparsity(self):
        return self.length / (self.space_size + 1E-10)

    def size(self, dim=None):
        if dim is None:
            return (*self.shape, self.channel)
        else:
            return (*self.shape, self.channel)[dim]

    def reduce_channel(self, reduction='max'):
        if reduction == 'max':
            return SparseTensor(torch.max(self.val, dim=-1, keepdim=True)[0], self.n, self.arity, 1, coo=self.coo)
        elif reduction == 'absmax':
            return SparseTensor(torch.max(self.val.abs(), dim=-1, keepdim=True)[0], self.n, self.arity, 1, coo=self.coo)
        elif reduction == 'sum':
            return SparseTensor(torch.sum(self.val, dim=-1, keepdim=True), self.n, self.arity, 1, coo=self.coo)
        elif reduction == 'mean':
            return SparseTensor(torch.mean(self.val, dim=-1, keepdim=True), self.n, self.arity, 1, coo=self.coo)
        elif reduction == 'min':
            return SparseTensor(torch.min(self.val, dim=-1, keepdim=True)[0], self.n, self.arity, 1, coo=self.coo)
        else:
            raise ValueError('Unknown reduction')

    def max(self):
        """[summary]
        Returns:
            SparseTensor: reduce the last dimension of coordinate space with max
        """
        if self.length == self.space_size:
            val = torch.max(self.val.view(
                *self.shape, self.channel), dim=-2)[0]
            return SparseTensor(val, self.n, self.arity - 1, self.channel)
        else:
            flatten_index = self.coo // self.n
            coo, inverse_index = torch.unique_consecutive(
                flatten_index, return_inverse=True)
            val = scatter(self.val, inverse_index, dim=0, reduce='max')
            return SparseTensor(val, self.n, self.arity - 1, self.channel, coo=coo)

    def min(self):
        """[summary]
        Returns:
            SparseTensor: reduce the last dimension of coordinate space with min
        """
        if self.length == self.space_size:
            val = torch.min(self.val.view(
                *self.shape, self.channel), dim=-2)[0]
            return SparseTensor(val, self.n, self.arity - 1, self.channel)
        else:
            flatten_index = self.coo // self.n
            coo, inverse_index = torch.unique_consecutive(
                flatten_index, return_inverse=True)
            val = scatter(self.val, inverse_index, dim=0, reduce='min')
            return SparseTensor(val, self.n, self.arity - 1, self.channel, coo=coo)

    def mean(self):
        """[summary]
        Returns:
            SparseTensor: reduce the last dimension of coordinate space with min
        """
        if self.length == self.space_size:
            val = torch.mean(self.val.view(*self.shape, self.channel), dim=-2)
            return SparseTensor(val, self.n, self.arity - 1, self.channel)
        else:
            flatten_index = self.coo // self.n
            coo, inverse_index = torch.unique_consecutive(
                flatten_index, return_inverse=True)
            val = scatter(self.val, inverse_index, dim=0, reduce='mean')
            return SparseTensor(val, self.n, self.arity - 1, self.channel, coo=coo)

    def sum(self):
        """[summary]
        Returns:
            SparseTensor: reduce the last dimension of coordinate space with min
        """
        if self.length == self.space_size:
            val = torch.sum(self.val.view(*self.shape, self.channel), dim=-2)
            return SparseTensor(val, self.n, self.arity - 1, self.channel)
        else:
            flatten_index = self.coo // self.n
            coo, inverse_index = torch.unique_consecutive(
                flatten_index, return_inverse=True)
            val = scatter(self.val, inverse_index, dim=0, reduce='sum')
            return SparseTensor(val, self.n, self.arity - 1, self.channel, coo=coo)

    @classmethod
    def cat(self, ts):
        nnzs = [t.length for t in ts]
        n = ts[0].n
        dim = ts[0].arity
        channels = [t.channel for t in ts]
        channel = sum(channels)
        vals = [t.val for t in ts]
        if nnzs.count(ts[0].space_size) == len(nnzs):
            coo = ts[0].coo
            val = torch.cat(vals, dim=-1)
            return SparseTensor(val, n, dim, channel)
        else:
            coo = torch.cat([t.coo for t in ts], dim=-1)
            coo, inverse_index = torch.unique(
                coo, return_inverse=True, sorted=True)
            val = torch.zeros((len(coo), channel),
                              dtype=ts[0].dtype, device=ts[0].device)

            def _cat(val, vals, nnzs, channels, inverse_index):
                current_channel, current_idx = 0, 0
                for t_val, t_len, t_c in zip(vals, nnzs, channels):
                    next_channel, next_idx = current_channel + t_c, current_idx + t_len
                    val[inverse_index[current_idx:next_idx],
                        current_channel:next_channel] = t_val
                    current_channel, current_idx = next_channel, next_idx

            _cat(val, vals, nnzs, channels, inverse_index)
            return SparseTensor(val, n, dim, channel, coo=coo)

    @functools.lru_cache(maxsize=1)
    def exclude_mask(self):
        mask = torch.zeros(self.length, device=self.device, dtype=torch.bool)
        coordinate = self.coordinate()
        for i in range(self.arity):
            for j in range(i+1, self.arity):
                mask = torch.logical_or(
                    mask, coordinate[:, i] == coordinate[:, j])
        return mask

    @functools.lru_cache(maxsize=1)
    def coordinate(self):
        coordinate = torch.zeros((self.length, self.arity),
                                 dtype=torch.long, device=self.device)
        coo = self.coo.clone()
        for i in reversed(range(self.arity)):
            coordinate[:, i] = coo % self.n
            coo //= self.n
        return coordinate

    def expand(self):
        coo = self.coo.clone()
        coo = (coo.unsqueeze(-1) * self.n) + torch.arange(self.n,
                                                          dtype=torch.long, device=self.device).unsqueeze(0)
        coo = coo.flatten()
        val = self.val.unsqueeze(1).expand(self.length, self.n, self.channel)
        val = val.contiguous().view(-1, self.channel)
        return SparseTensor(val, self.n, self.arity + 1, self.channel, coo=coo)

    def all_permutation(self):
        if self.arity <= 1:
            return self
        index = tuple(range(0, self.arity))
        ts = []
        if self.length == self.space_size:
            for p in itertools.permutations(index):
                val = self.val.view(*self.shape, self.channel)
                val = val.permute((*p, self.arity)).reshape((-1, self.channel))
                ts.append(SparseTensor(val, self.n,
                          self.arity, self.channel, coo=self.coo))
        else:
            weight = torch.tensor(
                [[self.n ** i for i in reversed(range(self.arity))]], dtype=torch.long, device=self.device)
            coordinate = self.coordinate()
            for p in itertools.permutations(index):
                new_coo = (weight * coordinate[:, p]).sum(-1)
                ts.append(SparseTensor(self.val, self.n,
                                       self.arity, self.channel, coo=new_coo))
        return self.cat(ts)

    def hoyer(self):
        if self.length in [0, 1]:
            return 0
        v = self.val.abs().max(-1)[0]
        l1 = v.sum() + 1e-10
        l2 = norm(v, ord=2) + 1e-10
        return (l1 / l2 - 1) / (self.length ** 0.5 - 1)

    def hoyer_square(self):
        if self.length in [0, 1]:
            return 1
        v = self.val.abs().max(-1)[0]
        return v.sum() ** 2 / (v ** 2).sum()

    def l1(self):
        if self.length < 1:
            return 0
        v = self.val.abs().max(-1)[0]
        l1 = v.sum()
        return l1

    def l2(self):
        if self.length < 1:
            return 0
        v = self.val.abs().max(-1)[0]
        l2 = (v ** 2).sum()
        return l2

    def l0(self):
        if self.length == 0:
            return 0
        return (self.val.abs().max(-1)[0] > EPSILON).float().mean()

    def __repr__(self):
        return f'SparseTensor(n={self.n}, dim={self.arity}, channel={self.channel}, \ncoo={self.coordinate()}, \nval={self.val})'

    def to(self, device):
        self.device = device
        self.val = self.val.to(device)
        self.coo = self.coo.to(device)
        return self
