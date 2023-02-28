import copy
import os.path as osp
from typing import List, Optional

import torch
from tqdm import tqdm
from torch_sparse import SparseTensor
from torch.utils.data.dataset import Dataset
import numpy as np
from numpy.linalg import matrix_power
from torch_sampling import choice
from .linkpred import AdjacencyList, find_all_path_nodes_c
from torch_geometric.utils.undirected import to_undirected
from torch_geometric.utils import negative_sampling
import random
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path


class SubgraphDataset(Dataset):
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
            flag = True
            while flag:
                self.data = self.dataset[idx // self.resample]
                self.data['relations'] = self.data['relations'].astype('float')
                if self.data['relations'].sum() > 0:
                    flag = False
                else:
                    idx += 1
            self.data_is_new = True
            self.cnt = 1
        else:
            self.data_is_new = False
            self.cnt += 1

        node_idx = self.__sample_nodes__(self.subgraph_size)

        data = copy.copy(self.data)
        data['n'] = node_idx.shape[0]
        for node_key in ['node_idx', 'node_feat', 'colors', 'states']:
            if node_key in data:
                data[node_key] = data[node_key][node_idx]
        data['relations'] = self.data['relations'][node_idx,
                                                   :, :][:, node_idx, :]
        if 'ternaries' in data and data['ternaries'] is not None:
            data['ternaries'] = data['ternaries'][node_idx,
                                                  :, :][:, node_idx, :][:, :, node_idx]
        self.target_dim = self.data['target'].ndim
        if self.data['target'].shape[-1] != self.data['n']:
            self.target_dim -= 1
        data['target'] = self.data['target']
        for dim in range(self.target_dim):
            idx = [slice(None)] * self.target_dim
            idx[dim] = node_idx
            data['target'] = data['target'][tuple(idx)]

        if self.do_calibrate:
            IS = self.get_IS(node_idx)
            if IS.ndim < data['target'].ndim:
                IS = IS[:, np.newaxis]
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


class SubgraphNodeDataset(SubgraphDataset):
    def __sample_nodes__(self, subgraph_size):
        node_idx = np.random.permutation(self.data['n'])[:subgraph_size]
        return node_idx


class SubgraphRandomWalkDataset(SubgraphDataset):
    def __init__(self, data, subgraph_size: int, start_num: int, walk_length: int,
                 epoch_size: int, resample: int = 10, IS_k: int = 2, IS_gamma: float = 0.3, do_calibrate: bool = False):
        self.walk_length = walk_length
        self.start_num = start_num
        super(SubgraphRandomWalkDataset,
              self).__init__(data, subgraph_size, epoch_size, resample, IS_k, IS_gamma, do_calibrate)

    def __sample_nodes__(self, subgraph_size):
        n = self.data['n']
        node_idx = torch.zeros(n, dtype=torch.bool)
        frontier = choice(torch.arange(n), self.start_num, False)
        node_idx.scatter_(0, frontier, 1)
        if self.data_is_new:
            self.ud_graph = self.data['relations'].max(axis=-1)
            self.ud_graph += self.ud_graph.T
            self.ud_graph *= 1 - np.identity(self.data['n'])
            self.ud_graph = SparseTensor.from_dense(
                torch.from_numpy(self.ud_graph), has_value=False)
            self.rowptr, self.col, _ = self.ud_graph.csr()
            self.rowcount = self.rowptr[1:] - self.rowptr[:-1]

        # print(self.rowptr)
        # print(self.rowcount)
        # print(0, ': ', frontier)
        for walk in range(self.walk_length):
            # print(walk+1)
            rnd = torch.rand(self.start_num, )
            # print(rnd)
            cnt = self.rowcount[frontier]
            # print(cnt)
            ptr = self.rowptr[frontier]
            # print(ptr)
            edge = torch.floor(rnd * cnt).long() + ptr
            edge[cnt == 0] = 0
            # print(edge)
            frontier = self.col[edge]
            node_idx.scatter_(0, frontier, 1)

        size = node_idx.sum()
        if size < subgraph_size:
            supplement = choice(
                (~node_idx).nonzero(as_tuple=True)[0], subgraph_size - size, False)
            node_idx[supplement] = True
        elif size > subgraph_size:
            return choice(node_idx.nonzero(as_tuple=True)[0], subgraph_size, False)
        return node_idx.nonzero(as_tuple=True)[0]


class SubgraphNeighborDataset(SubgraphDataset):
    def __init__(self, data, subgraph_size: int, neighbor_sizes: List[int],
                 epoch_size: int, resample: int = 10, IS_k: int = 2, IS_gamma: float = 0.3, do_calibrate: bool = False):
        self.neighbor_sizes = neighbor_sizes
        super(SubgraphNeighborDataset,
              self).__init__(data, subgraph_size, epoch_size, resample, IS_k, IS_gamma, do_calibrate)

    def __sample_nodes__(self, subgraph_size):
        n = self.data['n']
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
        # print(self.rowptr)
        # print(self.col)
        # print(0, ": ", next_hop)
        for k, neighbor_size in enumerate(self.neighbor_sizes[1:]):
            # print(k)
            cnt = self.rowcount[next_hop]
            # print(cnt)
            rnd = torch.rand(next_hop.size(0), neighbor_size)
            # print(rnd)
            ptr = self.rowptr[next_hop]
            # print(ptr)
            edge = torch.floor(rnd * cnt.unsqueeze(1)
                               ).long() + ptr.unsqueeze(1)
            # print(edge)
            edge[cnt == 0] = 0
            # print(edge)
            next_hop = torch.unique(self.col[edge].flatten())
            # print(next_hop)
            node_idx.scatter_(0, next_hop, 1)
        node_idx = node_idx.numpy()

        size = node_idx.sum()
        if size < subgraph_size:
            node_idx[np.random.permutation(
                (1 - node_idx).nonzero()[0])[:subgraph_size - size]] = 1
        elif size > subgraph_size:
            return np.random.permutation(node_idx.nonzero()[0])[:subgraph_size]
        return node_idx.nonzero()[0]


class SubgraphSingleDataset(SubgraphDataset):
    def __init__(self, data, subgraph_size: int, epoch_size: int, resample=50, k: int = 2, directed=True, bridge='rand', negative_sample=False, node_label=False):
        super(SubgraphSingleDataset, self).__init__(
            data, subgraph_size, epoch_size, resample, k, 0, False)
        self.directed = directed
        self.bridge = bridge
        self.negative_sample = negative_sample
        self.node_label = node_label
        self.k = k

    def __getitem__(self, idx):
        # {'n': N, 'relations': array(N, N, C), 'target': (N, N) }
        if self.cnt >= self.resample:
            flag = True
            while flag:
                self.data = self.dataset[idx // self.resample]
                self.data['relations'] = self.data['relations'].astype('float')
                if self.data['relations'].sum() > 0 and self.data['target'].sum() > 0:
                    flag = False
                else:
                    idx += 1
            self.data_is_new = True
            self.cnt = 1
        else:
            self.data_is_new = False
            self.cnt += 1

        node_idx, edge, target = self.__sample_nodes__(idx, self.subgraph_size)
        nodes = list(node_idx)
        edge[0], edge[1] = nodes.index(edge[0]), nodes.index(edge[1])

        data = copy.copy(self.data)
        data['n'] = node_idx.shape[0]
        for node_key in ['node_idx', 'node_feat', 'colors', 'states']:
            if node_key in data:
                data[node_key] = data[node_key][node_idx]
        data['relations'] = self.data['relations'][node_idx,
                                                   :, :][:, node_idx, :]

        data['target'] = target
        data['edge'] = edge
        data['node_idx'] = node_idx

        if self.node_label:
            graph = csr_matrix(data['relations'].sum(-1))
            dist_u = shortest_path(graph, directed=False, indices=edge[0])
            dist_v = shortest_path(graph, directed=False, indices=edge[1])
            data['node_feature'] = np.vstack([dist_u, dist_v]) + 1
            data['node_feature'][np.isinf(data['node_feature'])] = 0
        return data

    def __sample_nodes__(self, idx, subgraph_size):
        n = self.data['n']
        node_idx = torch.zeros(n, dtype=torch.bool)
        if self.data_is_new:
            self.n = self.data['n']
            self.edge = {"edge": torch.tensor(
                np.vstack(np.nonzero(self.data['target']))).T}
            self.num_rels = self.data['relations'].shape[-1]
            support_edges = torch.tensor(
                np.vstack(np.nonzero(self.data['relations'])))
            self.edge_index = torch.LongTensor(support_edges[:2, :])
            self.edge_type = torch.LongTensor(support_edges[2, :])
            self.adj = SparseTensor.from_edge_index(self.edge_index, torch.arange(
                self.edge_index.size(1)), sparse_sizes=(self.n, self.n))
            if self.directed:
                ud_edge_index = to_undirected(self.edge_index)
                ud_adj = SparseTensor.from_edge_index(
                    ud_edge_index, torch.arange(ud_edge_index.size(1)))
                self.adj_list = AdjacencyList(ud_adj)
            else:
                self.adj_list = AdjacencyList(self.adj)
            edge = self.edge["edge"].squeeze(-1)
            self.target_edge = torch.cat([edge, torch.ones(
                (edge.shape[0], 1), dtype=edge.dtype)], dim=1)
            if self.negative_sample:
                self.neg_edge = []

        if np.random.rand() < 0.5:
            edge = self.target_edge[np.random.choice(
                len(self.target_edge))].clone()
        else:
            while len(self.neg_edge) == 0:
                self.negative_sampling()
            edge = self.neg_edge[-1]
            self.neg_edge = self.neg_edge[:-1]

        if edge.size(0) == 4:
            edge, target = edge[:3], edge[3:].squeeze(-1)
        elif edge.size(0) == 3:
            edge, target = edge[:2], edge[2]
        else:
            print(edge)
            raise ValueError("edge size error")

        if self.bridge == 'rand':
            nodes = set(edge[:2].tolist())
        elif self.bridge == 'path':
            nodes = find_all_path_nodes_c(
                self.adj_list, edge[0], edge[1], k=self.k, subgraph_size=self.subgraph_size)
        else:
            raise NotImplementedError(
                "bridge {} not implemented".format(self.bridge))
        nodes = sorted(nodes)
        node_idx = torch.zeros(n, dtype=torch.int)
        node_idx.scatter_(0, torch.tensor(nodes), 1)
        node_idx = node_idx.numpy()
        size = node_idx.sum()

        if size < subgraph_size:
            node_idx[np.random.permutation(
                (1 - node_idx).nonzero()[0])[:subgraph_size - size]] = 1
        elif size > subgraph_size:
            return np.random.permutation(node_idx.nonzero()[0])[:subgraph_size]
        return node_idx.nonzero()[0], edge, target

    def negative_sampling(self):
        all_edge_index = self.target_edge[:, :2].t()
        if self.target_edge.size(1) == 3:
            neg_edge = negative_sampling(all_edge_index, self.n).t()
            self.neg_edge = torch.cat([neg_edge, torch.zeros(
                (neg_edge.shape[0], 1), dtype=neg_edge.dtype)], dim=1)
        elif self.target_edge.size(1) == 4:
            all_edge_type = torch.cat(
                [self.edge_type, self.target_edge[:, 2]], dim=0)
            neg_edges = []
            for r in range(self.num_rels):
                mask = all_edge_type == r
                neg_edge = [negative_sampling(
                    all_edge_index[:, mask], self.n)]
                size = neg_edge[0].size(1)
                neg_edge.append(torch.ones((1, size)).long() * r)
                neg_edge.append(torch.zeros((1, size)).long())
                neg_edge = torch.cat(neg_edge, dim=0)
                neg_edges.append(neg_edge.t())
            self.neg_edge = torch.cat(neg_edges, dim=0)
        else:
            raise NotImplementedError(
                "negative sampling for edge with size {} not implemented".format(self.target_edge.size(1)))


class SubgraphFullDataset(SubgraphDataset):
    def __sample_nodes__(self, subgraph_size):
        return torch.arange(self.N, dtype=torch.long)
