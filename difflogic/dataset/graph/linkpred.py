import pickle
from pandas.io import parsers
import torch
from torch.utils.data.dataset import Dataset
import numpy as np
from ogb.linkproppred import LinkPropPredDataset, PygLinkPropPredDataset
import torch_geometric.transforms as T
from torch_geometric.utils import num_nodes
from torch_geometric.utils.undirected import to_undirected
from torch_sparse.tensor import SparseTensor
from torch_geometric.utils import negative_sampling
import os.path as osp
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from . import GraphOutDegreeDataset, GraphConnectivityDataset, GraphAdjacentDataset, FamilyTreeDataset
from itertools import combinations
from .utils import sub_hypergraph, sample, flatten_hyper_edge_index, flatten_hyper_edges, construct_hyper_edge_index, construct_hyper_edges, hypergraph_negative_sampling, hypergraph_negative_sampling_with_edges, hypergraph_negative_sampling_with_flatten_edge_index, hypergraph_negative_sampling_with_flatten_edges, filter_from
from .path import AdjacencyList, find_all_path_nodes_c, find_all_path_nodes_py
from icecream import ic


class KnowledgeHyperGraph:
    def __init__(self, name, split, root='/afs/cs.stanford.edu/u/xgx/xgx_viscam/datasets/hyperkg'):
        self.root = root
        self.arity = int(name[-1])
        self._load_data(name, split)

    def _load_data(self, name, split):
        assert split in ['train', 'test']
        train_file = osp.join(self.root, name, 'train.txt')
        rel, ent = 0, 0
        rel2id, ent2id = {}, {}
        with open(train_file) as f:
            train_data = [line.split() for line in f.read().split('\n')[:-1]]
        for fact in train_data:
            if fact[0] not in rel2id:
                rel2id[fact[0]] = rel
                rel += 1

        support_data = train_data
        E = len(support_data)
        support_edges = torch.zeros(E, self.arity+1, dtype=torch.long)
        for idx, fact in enumerate(support_data):
            fact_list = []
            for entity in fact[1:]:
                if entity not in ent2id:
                    e_id = ent
                    ent2id[entity] = ent
                    ent += 1
                else:
                    e_id = ent2id[entity]
                fact_list.append(e_id)
            fact_list.append(rel2id[fact[0]])
            support_edges[idx] = torch.LongTensor(fact_list)
        if split == 'train':
            test_edges = support_edges
            e = E
        else:
            test_file = osp.join(self.root, name, 'test.txt')
            with open(test_file) as f:
                test_data = [line.split()
                             for line in f.read().split('\n')[:-1]]
            e = len(test_data)
            test_edges = torch.zeros(e, self.arity+1, dtype=torch.long)
            for idx, fact in enumerate(test_data):
                fact_list = []
                for entity in fact[1:]:
                    if entity not in ent2id:
                        e_id = ent
                        ent2id[entity] = ent
                        ent += 1
                    else:
                        e_id = ent2id[entity]
                    fact_list.append(e_id)
                fact_list.append(rel2id[fact[0]])
                test_edges[idx] = torch.LongTensor(fact_list)
        self.support_edges = support_edges
        self.test_edges = test_edges
        self.E = E
        self.e = e
        self.N = ent
        self.num_rels = rel


class KnowledgeGraph:
    def __init__(self, name, split, root='/afs/cs.stanford.edu/u/xgx/xgx_viscam/datasets/grail_data'):
        self.root = root
        self._load_data(name, split)

    def _load_data(self, name, split):
        assert split in ['train', 'test']
        train_file = osp.join(self.root, name, 'train.txt')
        rel, ent = 0, 0
        rel2id, ent2id = {}, {}
        with open(train_file) as f:
            train_data = [line.split() for line in f.read().split('\n')[:-1]]
        for triplet in train_data:
            if triplet[1] not in rel2id:
                rel2id[triplet[1]] = rel
                rel += 1

        support_data = train_data
        E = len(support_data)
        support_edges = torch.zeros(E, 3, dtype=torch.long)
        for idx, triplet in enumerate(support_data):
            if triplet[0] not in ent2id:
                u = ent
                ent2id[triplet[0]] = ent
                ent += 1
            else:
                u = ent2id[triplet[0]]
            if triplet[2] not in ent2id:
                v = ent
                ent2id[triplet[2]] = ent
                ent += 1
            else:
                v = ent2id[triplet[2]]
            r = rel2id[triplet[1]]
            support_edges[idx] = torch.LongTensor([u, v, r])
        if split == 'train':
            test_edges = support_edges
            e = E
        else:
            test_file = osp.join(self.root, name, 'test.txt')
            with open(test_file) as f:
                test_data = [line.split()
                             for line in f.read().split('\n')[:-1]]
            e = len(test_data)
            test_edges = torch.zeros(e, 3, dtype=torch.long)
            for idx, triplet in enumerate(test_data):
                if triplet[0] not in ent2id:
                    u = ent
                    ent2id[triplet[0]] = ent
                    ent += 1
                else:
                    u = ent2id[triplet[0]]
                if triplet[2] not in ent2id:
                    v = ent
                    ent2id[triplet[2]] = ent
                    ent += 1
                else:
                    v = ent2id[triplet[2]]
                r = rel2id[triplet[1]]
                test_edges[idx] = torch.LongTensor([u, v, r])
        self.support_edges = support_edges
        self.test_edges = test_edges
        self.E = E
        self.e = e
        self.N = ent
        self.num_rels = rel


class GrailGraph:
    def __init__(self, name, split, root='/afs/cs.stanford.edu/u/xgx/xgx_viscam/datasets/grail_data'):
        self.root = root
        self._load_data(name, split)

    def _load_data(self, name, split):
        assert split in ['train', 'test']
        assert '_ind' not in name
        train_file = osp.join(self.root, name, 'train.txt')
        rel, ent = 0, 0
        rel2id, ent2id = {}, {}
        with open(train_file) as f:
            train_data = [line.split() for line in f.read().split('\n')[:-1]]
        for triplet in train_data:
            if triplet[1] not in rel2id:
                rel2id[triplet[1]] = rel
                rel += 1

        if split == 'test':
            name += '_ind'
        support_file = osp.join(self.root, name, 'train.txt')
        with open(support_file) as f:
            support_data = [line.split() for line in f.read().split('\n')[:-1]]
        E = len(support_data)
        support_edges = torch.zeros(E, 3, dtype=torch.long)
        for idx, triplet in enumerate(support_data):
            if triplet[0] not in ent2id:
                u = ent
                ent2id[triplet[0]] = ent
                ent += 1
            else:
                u = ent2id[triplet[0]]
            if triplet[2] not in ent2id:
                v = ent
                ent2id[triplet[2]] = ent
                ent += 1
            else:
                v = ent2id[triplet[2]]
            r = rel2id[triplet[1]]
            support_edges[idx] = torch.LongTensor([u, v, r])

        test_file = osp.join(self.root, name, 'test.txt')
        with open(test_file) as f:
            test_data = [line.split() for line in f.read().split('\n')[:-1]]
        e = len(test_data)
        test_edges = torch.zeros(e, 3, dtype=torch.long)
        for idx, triplet in enumerate(test_data):
            if triplet[0] not in ent2id:
                u = ent
                ent2id[triplet[0]] = ent
                ent += 1
            else:
                u = ent2id[triplet[0]]
            if triplet[2] not in ent2id:
                v = ent
                ent2id[triplet[2]] = ent
                ent += 1
            else:
                v = ent2id[triplet[2]]
            r = rel2id[triplet[1]]
            test_edges[idx] = torch.LongTensor([u, v, r])
        self.support_edges = support_edges
        self.test_edges = test_edges
        self.E = E
        self.e = e
        self.N = ent
        self.num_rels = rel


class LinkPredDataset(Dataset):
    def __init__(self, name, split):
        super(LinkPredDataset, self).__init__()
        self.name = name
        self.split = split
        if name[:3] == 'ogb':
            self._load_ogb(name, split)
        elif name[:6] == 'grail-':
            self._load_grail(name[6:], split)
        elif name[:3] == 'kg-':
            self._load_kg(name[3:], split)
        elif name[:4] == 'hkg-':
            self._load_hyperkg(name[4:], split)
        else:
            raise NotImplementedError

    def _load_ogb(self, name, split):
        self.dataset = PygLinkPropPredDataset(
            name, root='/afs/cs.stanford.edu/u/xgx/xgx_viscam/datasets')
        self.edge = self.dataset.get_edge_split()[split]
        graph = self.dataset[0]
        print(graph)
        self.n = graph['num_nodes']
        self.edge_index = graph['edge_index']

        if 'edge_weight' in graph:
            self.edge_weight = graph['edge_weight']
        self.num_rels = 1
        self.node_feature = graph['x']
        self.arity = 2

    def _load_grail(self, name, split):
        graph = GrailGraph(name, split)
        self.n = graph.N
        self.edge = {"edge": graph.test_edges}
        self.num_rels = graph.num_rels
        self.edge_index = graph.support_edges[:, :2].T
        self.edge_type = graph.support_edges[:, 2]
        self.node_feature = None
        self.arity = 2

    def _load_kg(self, name, split):
        graph = KnowledgeGraph(name, split)
        print(
            f'{name}-{split}: N = {graph.N}, E = {graph.E}, e = {graph.e}, R = {graph.num_rels}')
        self.n = graph.N
        self.edge = {"edge": graph.test_edges}
        self.num_rels = graph.num_rels
        self.edge_index = graph.support_edges[:, :2].T
        self.edge_type = graph.support_edges[:, 2]
        self.node_feature = None
        self.arity = 2

    def _load_hyperkg(self, name, split):
        graph = KnowledgeHyperGraph(name, split)
        print(
            f'{name}-{split}: N = {graph.N}, E = {graph.E}, e = {graph.e}, R = {graph.num_rels}')
        self.n = graph.N
        self.edge = {"edge": graph.test_edges}
        self.num_rels = graph.num_rels
        self.arity = graph.arity
        comb = combinations(range(self.arity), 2)
        self.edge_index = []
        for idx in comb:
            self.edge_index.append(graph.support_edges[:, idx].T)
        self.edge_index = torch.cat(self.edge_index, dim=1)
        self.hyper_edge_index = graph.support_edges[:, :-1]
        if split == 'train':
            self.all_edges = graph.test_edges
        else:
            self.all_edges = torch.cat(
                [graph.test_edges, graph.support_edges], dim=0)
        self.flatten_all_edges = flatten_hyper_edges(
            self.all_edges, self.arity, self.n)
        self.edge_type = graph.support_edges[:, self.arity]
        self.node_feature = None

    def __len__(self):
        return 1

    def __getitem__(self, item):
        if not hasattr(self, 'relations'):
            self.relations = np.zeros((self.n, self.n, 1))
            self.relations[self.edge_index[0], self.edge_index[1], :] = 1
            if hasattr(self, 'edge_weight'):
                edge_feat = np.zeros(
                    (self.n, self.n, self.edge_weight.shape[-1]))
                edge_feat[self.edge_index[0],
                          self.edge_index[1], :] = self.edge_weight
                self.relations = np.concatenate(
                    (self.relations, edge_feat), axis=-1)
        if not hasattr(self, 'target'):
            self.target = np.zeros((self.n, self.n))
            self.target[self.edge["edge"][:, 0], self.edge["edge"][:, 1]] = 1
        ret_dict = dict(n=self.n, relations=self.relations,
                        target=self.target, node_idx=torch.arange(self.n))
        if self.node_feature is not None:
            ret_dict['node_feature'] = self.node_feature
        return ret_dict


class SingleLinkPredDataset(LinkPredDataset):
    def __init__(self, name, split, k=2, subgraph_size=-1, directed=True, bridge='rand', negative_sample=False, node_label=False):
        super(SingleLinkPredDataset, self).__init__(name, split)
        self.node_label = node_label
        self.adj = SparseTensor.from_edge_index(
            self.edge_index, torch.arange(self.edge_index.size(1)), sparse_sizes=(self.n, self.n))
        if directed:
            ud_edge_index = to_undirected(self.edge_index)
            ud_adj = SparseTensor.from_edge_index(
                ud_edge_index, torch.arange(ud_edge_index.size(1)), sparse_sizes=(self.n, self.n))
            self.adj_list = AdjacencyList(ud_adj)
        else:
            self.adj_list = AdjacencyList(self.adj)

        edge = self.edge["edge"].squeeze(-1)
        self.target_edge = torch.cat([edge, torch.ones(
            (edge.shape[0], 1), dtype=edge.dtype)], dim=1)
        self.k = k
        self.subgraph_size = subgraph_size
        self.bridge = bridge
        self.negative_sample = negative_sample
        if "edge_neg" in self.edge:
            neg_edge = self.edge["edge_neg"]
            neg_edge = torch.cat([neg_edge, torch.zeros(
                (neg_edge.shape[0], 1), dtype=neg_edge.dtype)], dim=1)
            self.target_edge = torch.cat([self.target_edge, neg_edge], dim=0)
            self.negative_sample = False
        if self.negative_sample:
            self.neg_edge = []

    def __len__(self):
        if self.negative_sample:
            return self.target_edge.shape[0] * 2
        return self.target_edge.shape[0]

    def test_negative_edges(self, edge, miss_domain):
        # edge = [u_1, ..., u_r, r]
        neg_edges = edge.repeat(self.n, 1)
        neg_edges[:, miss_domain] = torch.arange(self.n)
        flatten_neg_edges = flatten_hyper_edges(neg_edges, self.arity, self.n)
        flatten_neg_edges = filter_from(
            flatten_neg_edges, self.flatten_all_edges)
        neg_edges = construct_hyper_edges(
            flatten_neg_edges, self.arity, self.n)
        return neg_edges

    def construct_subgraph(self, edge):
        edge = edge.clone()
        edge, target = edge[:-1], edge[-1]
        nodes = set(edge[:self.arity].tolist())
        if self.bridge == 'path':
            comb = list(combinations(edge[:self.arity], 2))
            comb_num = len(comb)
            for e in comb:
                nodes |= find_all_path_nodes_c(
                    self.adj_list, e[0], e[1], k=self.k, subgraph_size=self.subgraph_size // comb_num)
        self.supplement_nodes(nodes)
        nodes = sorted(nodes)
        for i in range(self.arity):
            edge[i] = nodes.index(edge[i])
        nodes = torch.tensor(nodes)

        relations, ternaries = None, None
        if self.arity == 2:
            adj, _ = self.adj.saint_subgraph(nodes)
            row, col, edge_idx = adj.coo()
            if self.num_rels == 1:
                relations = np.zeros(
                    (self.subgraph_size, self.subgraph_size, 1))
                relations[row, col, :] = 1
                if self.split == 'train':
                    relations[edge[0], edge[1], :] = 0
            else:
                relations = np.zeros(
                    (self.subgraph_size, self.subgraph_size, self.num_rels))
                relations[row, col, self.edge_type[edge_idx]] = 1
                if self.split == 'train':
                    relations[edge[0], edge[1], edge[2]] = 0

            if hasattr(self, 'edge_weight'):
                edge_feat = np.zeros(
                    (self.subgraph_size, self.subgraph_size, self.edge_weight.shape[-1]))
                edge_feat[row, col, :] = self.edge_weight[edge_idx]
                relations = np.concatenate(
                    (relations, edge_feat), axis=-1)

        elif self.arity == 3:
            new_edge_index, edge_idx = sub_hypergraph(
                nodes, self.hyper_edge_index, self.n)
            ternaries = np.zeros(
                (self.subgraph_size, self.subgraph_size, self.subgraph_size, self.num_rels))
            ternaries[new_edge_index[:, 0], new_edge_index[:, 1],
                      new_edge_index[:, 2], self.edge_type[edge_idx]] = 1
            if self.split == 'train':
                ternaries[edge[0], edge[1], edge[2], edge[3]] = 0
        else:
            raise NotImplementedError
        ret_dict = dict(n=self.subgraph_size, edge=edge,
                        target=target, node_idx=nodes)
        if relations is not None:
            ret_dict['relations'] = relations
        if ternaries is not None:
            ret_dict['ternaries'] = ternaries

        if self.node_label:
            graph = csr_matrix(relations.sum(-1))
            dist_u = shortest_path(graph, directed=False, indices=edge[0])
            dist_v = shortest_path(graph, directed=False, indices=edge[1])
            ret_dict['node_feature'] = torch.vstack([dist_u, dist_v])
        elif self.node_feature is not None:
            ret_dict['node_feature'] = self.node_feature[nodes]

        return ret_dict

    def __getitem__(self, idx):
        # TODO: SAVE A SPLIT FILE SO THAT WE DON'T HAVE TO REPEAT THIS
        # Deciding whether to sample from the negative edges
        if idx < self.target_edge.shape[0]:
            edge = self.target_edge[idx].clone()
        elif self.negative_sample:
            while len(self.neg_edge) == 0:
                self.negative_sampling()
            edge = self.neg_edge[-1]
            self.neg_edge = self.neg_edge[:-1]
        else:
            raise IndexError

        # When doing multi-relational link prediction, the edge is (*edge_seq, rel, exist-or-not)
        # When doing single-relational link prediction, the edge is (*edge_seq, exist-or-not)
        return self.construct_subgraph(edge)

    def supplement_nodes(self, nodes):
        while len(nodes) < self.subgraph_size:
            new_node = np.random.choice(self.n)
            nodes.add(new_node)

    def negative_sampling(self):
        if self.arity > 2:
            if self.split == 'train':
                size = self.target_edge.shape[0]
            elif self.split == 'test':
                size = self.target_edge.shape[0]
            neg_edges = hypergraph_negative_sampling_with_flatten_edges(
                self.flatten_all_edges, self.n, self.arity, self.num_rels, size)
            neg_labels = torch.zeros(size, 1, dtype=torch.long)
            self.neg_edge = torch.cat(
                [neg_edges, neg_labels], dim=1)
            return
        all_edge_index = torch.cat(
            [self.edge_index, self.target_edge[:, :2].t()], dim=1)
        if self.target_edge.size(1) == self.arity + 1:
            # Binary link prediction
            neg_edge = negative_sampling(all_edge_index, self.n).t()
            self.neg_edge = torch.cat([neg_edge, torch.zeros(
                (neg_edge.shape[0], 1), dtype=neg_edge.dtype)], dim=1)
        elif self.target_edge.size(1) == self.arity + 2:
            # Multi-relational link prediction
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


class SingleContrastiveLinkPredDataset(LinkPredDataset):
    def __init__(self, name, split, num_neg_per_pos=1, k=2, subgraph_size=-1, directed=True, bridge='rand', node_label=False, shuffle=False):
        super(SingleContrastiveLinkPredDataset, self).__init__(name, split)
        self.node_label = node_label
        self.adj = SparseTensor.from_edge_index(
            self.edge_index, torch.arange(self.edge_index.size(1)), sparse_sizes=(self.n, self.n))
        if directed:
            ud_edge_index = to_undirected(self.edge_index)
            ud_adj = SparseTensor.from_edge_index(
                ud_edge_index, torch.arange(ud_edge_index.size(1)), sparse_sizes=(self.n, self.n))
            self.adj_list = AdjacencyList(ud_adj)
        else:
            self.adj_list = AdjacencyList(self.adj)

        edge = self.edge["edge"].squeeze(-1)
        self.target_edge = torch.cat([edge, torch.ones(
            (edge.shape[0], 1), dtype=edge.dtype)], dim=1)
        self.k = k
        self.subgraph_size = subgraph_size
        self.bridge = bridge
        self.num_neg_per_pos = num_neg_per_pos
        if shuffle:
            self.target_edge = self.target_edge[torch.randperm(
                self.target_edge.shape[0])]
        self.neg_edge = []

    def __len__(self):
        return self.target_edge.shape[0] * (1 + self.num_neg_per_pos)

    def construct_subgraph(self, edge):
        edge = edge.clone()
        edge, target = edge[:-1], edge[-1]
        nodes = set(edge[:self.arity].tolist())
        if self.bridge == 'path':
            comb = list(combinations(edge[:self.arity], 2))
            comb_num = len(comb)
            for e in comb:
                nodes |= find_all_path_nodes_c(
                    self.adj_list, e[0], e[1], k=self.k, subgraph_size=self.subgraph_size // comb_num)
        self.supplement_nodes(nodes)
        nodes = sorted(nodes)
        for i in range(self.arity):
            edge[i] = nodes.index(edge[i])
        nodes = torch.tensor(nodes)

        relations, ternaries = None, None
        if self.arity == 2:
            adj, _ = self.adj.saint_subgraph(nodes)
            row, col, edge_idx = adj.coo()
            if self.num_rels == 1:
                relations = np.zeros(
                    (self.subgraph_size, self.subgraph_size, 1))
                relations[row, col, :] = 1
                if self.split == 'train':
                    relations[edge[0], edge[1], :] = 0
            else:
                relations = np.zeros(
                    (self.subgraph_size, self.subgraph_size, self.num_rels))
                relations[row, col, self.edge_type[edge_idx]] = 1
                if self.split == 'train':
                    relations[edge[0], edge[1], edge[2]] = 0

            if hasattr(self, 'edge_weight'):
                edge_feat = np.zeros(
                    (self.subgraph_size, self.subgraph_size, self.edge_weight.shape[-1]))
                edge_feat[row, col, :] = self.edge_weight[edge_idx]
                relations = np.concatenate(
                    (relations, edge_feat), axis=-1)

        elif self.arity == 3:
            new_edge_index, edge_idx = sub_hypergraph(
                nodes, self.hyper_edge_index, self.n)
            ternaries = np.zeros(
                (self.subgraph_size, self.subgraph_size, self.subgraph_size, self.num_rels))
            ternaries[new_edge_index[:, 0], new_edge_index[:, 1],
                      new_edge_index[:, 2], self.edge_type[edge_idx]] = 1
            if self.split == 'train':
                ternaries[edge[0], edge[1], edge[2], edge[3]] = 0
        else:
            raise NotImplementedError
        ret_dict = dict(n=self.subgraph_size, edge=edge,
                        target=target, node_idx=nodes)
        if relations is not None:
            ret_dict['relations'] = relations
        if ternaries is not None:
            ret_dict['ternaries'] = ternaries

        if self.node_label:
            graph = csr_matrix(relations.sum(-1))
            dist_u = shortest_path(graph, directed=False, indices=edge[0])
            dist_v = shortest_path(graph, directed=False, indices=edge[1])
            ret_dict['node_feature'] = torch.vstack([dist_u, dist_v])
        elif self.node_feature is not None:
            ret_dict['node_feature'] = self.node_feature[nodes]

        return ret_dict

    def __getitem__(self, idx):
        # When doing multi-relational link prediction, the edge is (*edge_seq, rel, exist-or-not)
        # When doing single-relational link prediction, the edge is (*edge_seq, exist-or-not)
        idx, remainder = divmod(idx, self.num_neg_per_pos + 1)
        if remainder == 0:
            edge = self.target_edge[idx].clone()
        else:
            edge = self.replace_negative_sample(self.target_edge[idx].clone())
            # while len(self.neg_edge) == 0:
            #     self.rand_negative_sampling()
            # edge = self.neg_edge[-1]
            # self.neg_edge = self.neg_edge[:-1]
        return self.construct_subgraph(edge)

    def supplement_nodes(self, nodes):
        while len(nodes) < self.subgraph_size:
            new_node = np.random.choice(self.n)
            nodes.add(new_node)

    def replace_negative_sample(self, edge):
        arity_to_replace = np.random.randint(self.arity)
        edge[arity_to_replace] = np.random.choice(self.n)
        edge[-1] = 0
        return edge

    def rand_negative_sampling(self):
        all_edge_index = torch.cat(
            [self.edge_index, self.target_edge[:, :2].t()], dim=1)
        if self.target_edge.size(1) == self.arity + 1:
            # Binary link prediction
            neg_edge = negative_sampling(all_edge_index, self.n).t()
            self.neg_edge = torch.cat([neg_edge, torch.zeros(
                (neg_edge.shape[0], 1), dtype=neg_edge.dtype)], dim=1)
        elif self.target_edge.size(1) == self.arity + 2:
            # Multi-relational link prediction
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


def test_single_link_pred():
    from time import time
    from jactorch.data.dataloader import JacDataLoader

    dataset = SingleLinkPredDataset(
        'ogbl-ddi', 'train', k=2, subgraph_size=40, directed=False)
    dataloader = JacDataLoader(
        dataset,
        shuffle=False,
        batch_size=200,
        num_workers=8)
    iterator = dataloader.__iter__()
    test_num = 20
    test_time = 0
    for i in range(test_num):
        start = time()
        data = iterator.next()
        print(data["edge"])
        end = time()
        test_time += end - start
    print(f"Mean test time: {test_time / test_num}")


def precompute_nodes():
    dataset = SingleLinkPredDataset(
        'ogbl-ddi', 'test', k=2, subgraph_size=100, directed=False)
    dataset.precompute_nodes()
    dataset.subgraph_size = 50
    dataset.precompute_nodes()


def test_collab():
    from time import time
    from jactorch.data.dataloader import JacDataLoader
    dataset = SingleLinkPredDataset(
        'ogbl-collab', 'train', k=3, subgraph_size=40, directed=False, bridge='path')
    dataloader = JacDataLoader(
        dataset,
        shuffle=False,
        batch_size=1,
        num_workers=0,
        collate_fn=dataset.collate_fn)
    iterator = dataloader.__iter__()
    test_num = 10
    test_time = 0
    for i in range(test_num):
        start = time()
        data = iterator.next()
        end = time()
        test_time += end - start
    print(f"Mean test time: {test_time / test_num}")


def summary(arr):
    arr = np.array(arr)
    print(arr.mean(), arr.std(), arr.min(), arr.max())


def test_grail():
    from time import time
    from jactorch.data.dataloader import JacDataLoader
    dataset = SingleLinkPredDataset(
        'grail-WN18RR_v4_ind', 'train', k=3, subgraph_size=20, directed=True, bridge='path', negative_sample=True)
    dataloader = JacDataLoader(
        dataset,
        shuffle=True,
        batch_size=1,
        num_workers=0)
    iterator = dataloader.__iter__()
    test_num = 2000
    test_time = 0
    cnt1, cnt2 = [], []
    for i in range(test_num):
        start = time()
        data = iterator.next()
        target = data["target"]
        sm = data["n_nodes"].item()
        if target == 1:
            cnt1.append(sm)
        else:
            cnt2.append(sm)
        end = time()
        test_time += end - start
    summary(cnt1)
    summary(cnt2)

    print(f"Mean test time: {test_time / test_num}")


if __name__ == '__main__':
    # test_single_link_pred()
    # test_collab()
    test_grail()
    # precompute_nodes()
