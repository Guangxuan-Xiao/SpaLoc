import copy
import os.path as osp
from typing import List, Optional

import torch
from torch.autograd.grad_mode import F
from tqdm import tqdm
from torch_sparse import SparseTensor
from torch.utils.data.dataset import Dataset
import numpy as np
from numpy.linalg import matrix_power
from torch_sampling import choice


class SparseSubgraphDataset(Dataset):
    def __init__(self, dataset, subgraph_size: int, epoch_size: int, resample: int = 50, IS_k: int = 2, IS_gamma: float = 0.3, do_calibrate: bool = False):
        self.subgraph_size = subgraph_size
        self.dataset = dataset
        self.epoch_size = epoch_size
        self.IS_k = IS_k
        self.resample = resample
        self.cnt = self.resample
        self.do_calibrate = do_calibrate
        self.IS_gamma = IS_gamma

    def __sample_nodes__(self, subgraph_size):
        raise NotImplementedError

    def __getitem__(self, idx):
        # {'n': N, 'relations': array(N, N, C), 'target': (N, N) }
        if self.cnt >= self.resample:
            self.data = self.dataset[idx // self.resample]
            self.data_is_new = True
            self.cnt = 1
        else:
            self.data_is_new = False
            self.cnt += 1

        node_idx = self.__sample_nodes__(self.subgraph_size)

        data = copy.copy(self.data)
        N, E = data['num_nodes'], data['edge_index'].size(1)
        adj = SparseTensor.from_edge_index(
            data['edge_index'], sparse_sizes=(N, N))
        adj, edge_idx = adj.saint_subgraph(node_idx)
        data['num_nodes'] = node_idx.shape[0]
        for k, v in data.items():
            if not torch.is_tensor(v):
                continue
            if v.size(0) == N:
                data[k] = v[node_idx]
            elif v.size(0) == E:
                data[k] = v[edge_idx]
        row, col, _ = adj.coo()
        data['edge_index'] = torch.stack([row, col], dim=0)

        self.target_dim = self.data['target'].ndim
        data['target'] = self.data['target']
        for dim in range(self.target_dim):
            idx = [slice(None)] * self.target_dim
            idx[dim] = node_idx
            data['target'] = data['target'][tuple(idx)]

        if self.do_calibrate:
            IS = self.get_IS(node_idx)
            data['target'] *= IS
        return data

    def get_IS(self, node_idx):
        n = node_idx.shape[0]
        if self.data_is_new:
            self.graph = self.data['relations'].sum(axis=-1)
            self.graph += self.graph.T
            self.N_tot = np.identity(self.data['n'])
            for k in range(self.IS_k):
                self.N_tot += matrix_power(self.graph, k+1)
        subgraph = self.graph[node_idx, :][:, node_idx].copy()
        n_sub = np.identity(n)
        for k in range(self.IS_k):
            n_sub += matrix_power(subgraph, k+1)
        if self.target_dim == 2:
            n_tot = self.N_tot[node_idx, :][:, node_idx]
        elif self.target_dim == 1:
            n_sub = n_sub.sum(-1)
            n_tot = self.N_tot[node_idx, :][:, node_idx].sum(-1)
        IS = n_sub / n_tot
        IS = np.where(np.isnan(IS), 1, IS) ** self.IS_gamma
        return IS

    def calc_MIS(self, idx):
        self.data = self.dataset[idx]
        mis_list = []
        graph = self.data['relations'].sum(axis=-1)
        diag = np.identity(self.data['n'])
        graph = graph * (1 - diag) + diag
        N_tot = matrix_power(graph, self.IS_k)
        for i in range(self.resample):
            self.data_is_new = i == 0
            node_idx = self.__sample_nodes__(self.subgraph_size)
            n = node_idx.shape[0]
            subgraph = graph[node_idx, :][:, node_idx].copy()
            n_sub = matrix_power(subgraph, self.IS_k)
            n_tot = N_tot[node_idx, :][:, node_idx]
            diag = np.identity(n)
            n_sub *= (1 - diag)
            n_tot *= (1 - diag)
            if n_tot.sum() == 0:
                mis_list.append(0)
                continue
            idxs = n_tot.nonzero()
            mis = (n_sub[idxs] / n_tot[idxs]).mean()
            mis_list.append(mis)
        return mis_list

    def __len__(self):
        return self.epoch_size


class SparseSubgraphNeighborDataset(SparseSubgraphDataset):
    def __init__(self, data, subgraph_size: int, neighbor_sizes: List[int],
                 epoch_size: int, resample: int = 10, IS_k: int = 2, IS_gamma: float = 0.3, do_calibrate: bool = False):
        self.neighbor_sizes = neighbor_sizes
        super(SparseSubgraphNeighborDataset,
              self).__init__(data, subgraph_size, epoch_size, resample, IS_k, IS_gamma, do_calibrate)

    def __sample_nodes__(self, subgraph_size):
        n = self.data['num_nodes']
        node_idx = torch.zeros(n, dtype=torch.bool)
        # next_hop = torch.randperm(n)[:self.neighbor_sizes[0]]
        next_hop = choice(torch.arange(n), self.neighbor_sizes[0], False)
        node_idx.scatter_(0, next_hop, 1)
        if self.data_is_new:
            self.ud_graph = self.data['relations'].max(axis=-1)
            self.ud_graph += self.ud_graph.T
            self.ud_graph *= 1 - np.identity(n)
            self.ud_graph = SparseTensor.from_dense(
                torch.from_numpy(self.ud_graph), has_value=False)
            self.rowptr, self.col, _ = self.ud_graph.csr()
            self.rowcount = self.rowptr[1:] - self.rowptr[:-1]
        for k, neighbor_size in enumerate(self.neighbor_sizes[1:]):
            cnt = self.rowcount[next_hop]
            rnd = torch.rand(next_hop.size(0), neighbor_size)
            ptr = self.rowptr[next_hop]
            edge = torch.floor(rnd * cnt.unsqueeze(1))
            edge[cnt == 0] = 0
            next_hop = torch.unique(self.col[edge].flatten())
            node_idx.scatter_(0, next_hop, 1)
        node_idx = node_idx.numpy()

        size = node_idx.sum()
        if size < subgraph_size:
            node_idx[np.random.permutation(
                (1 - node_idx).nonzero()[0])[:subgraph_size - size]] = 1
        elif size > subgraph_size:
            return np.random.permutation(node_idx.nonzero()[0])[:subgraph_size]
        return node_idx.nonzero()[0]
