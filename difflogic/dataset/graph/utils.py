import random
import torch
import numpy as np


def sub_hypergraph(node_idx, edge_index, n):
    node_map = torch.full((n,), -1, dtype=torch.long)
    node_map.index_copy_(0, node_idx, torch.arange(len(node_idx)))
    new_edge_index = node_map[edge_index]
    edge_idx = torch.all(new_edge_index >= 0, dim=1)
    new_edge_index = new_edge_index[edge_idx]
    return new_edge_index, edge_idx


def sample(high: int, size: int, device=None):
    size = min(high, size)
    return torch.tensor(random.sample(range(high), size), device=device)


def flatten_hyper_edge_index(edge_index, arity, n):
    weight = torch.tensor([[n**i for i in range(arity)]], dtype=torch.long)
    return (edge_index@weight.T).squeeze()


def flatten_hyper_edges(edges, arity, n):
    weight = torch.tensor([[n**i for i in range(arity + 1)]], dtype=torch.long)
    return (edges@weight.T).squeeze()


def construct_hyper_edge_index(flatten_edge_index, arity, n):
    flatten_edge_index = flatten_edge_index.clone()
    edge_index = torch.empty(
        (flatten_edge_index.shape[0], arity), dtype=torch.long)
    for i in range(arity):
        edge_index[:, i] = flatten_edge_index % n
        flatten_edge_index = flatten_edge_index // n
    return edge_index


def construct_hyper_edges(flatten_edges, arity, n):
    flatten_edges = flatten_edges.clone()
    edges = torch.empty((flatten_edges.shape[0], arity + 1), dtype=torch.long)
    for i in range(arity + 1):
        edges[:, i] = flatten_edges % n
        flatten_edges = flatten_edges // n
    return edges


def hypergraph_negative_sampling(edge_index, n, num_neg_samples=None):
    E, arity = edge_index.shape
    flatten_edge_index = flatten_hyper_edge_index(edge_index, arity, n)
    num_neg_samples = num_neg_samples or E
    size = n ** arity
    num_neg_samples = min(num_neg_samples, size - E)
    alpha = abs(1 / (1 - 1.1 * (E / size)))
    perm = sample(size, int(alpha * num_neg_samples))
    mask = torch.from_numpy(
        np.isin(perm, flatten_edge_index.numpy())).to(torch.bool)
    flatten_neg_edge_index = perm[~mask][:num_neg_samples]
    neg_edge_index = construct_hyper_edge_index(
        flatten_neg_edge_index, arity, n)
    return neg_edge_index


def hypergraph_negative_sampling_with_edges(edges, n, r, num_neg_samples=None):
    E, arity = edges.shape
    arity -= 1
    flatten_edges = flatten_hyper_edges(edges, arity, n)
    num_neg_samples = num_neg_samples or E
    size = r * n ** arity
    num_neg_samples = min(num_neg_samples, size - E)
    alpha = abs(1 / (1 - 1.1 * (E / size)))
    perm = sample(size, int(alpha * num_neg_samples))
    mask = torch.from_numpy(
        np.isin(perm, flatten_edges.numpy())).to(torch.bool)
    flatten_neg_edges = perm[~mask][:num_neg_samples]
    neg_edges = construct_hyper_edges(flatten_neg_edges, arity, n)
    return neg_edges

def filter_from(current, avoid):
    mask = (current[..., None] == avoid).any(-1)
    return current[~mask]

def hypergraph_negative_sampling_with_flatten_edge_index(flatten_edge_index, n, arity, num_neg_samples=None):
    E = flatten_edge_index.shape[0]
    num_neg_samples = num_neg_samples or E
    size = n ** arity
    num_neg_samples = min(num_neg_samples, size - E)
    alpha = abs(1 / (1 - 1.1 * (E / size)))
    perm = sample(size, int(alpha * num_neg_samples))
    mask = torch.from_numpy(
        np.isin(perm, flatten_edge_index.numpy())).to(torch.bool)
    flatten_neg_edge_index = perm[~mask][:num_neg_samples]
    neg_edge_index = construct_hyper_edge_index(
        flatten_neg_edge_index, arity, n)
    return neg_edge_index

def hypergraph_negative_sampling_with_flatten_edges(flatten_edges, n, arity, r, num_neg_samples=None):
    E = flatten_edges.shape[0]
    num_neg_samples = num_neg_samples or E
    size = r * n ** arity
    num_neg_samples = min(num_neg_samples, size - E)
    alpha = abs(1 / (1 - 1.1 * (E / size)))
    perm = sample(size, int(alpha * num_neg_samples))
    mask = torch.from_numpy(np.isin(perm, flatten_edges.numpy())).to(torch.bool)
    flatten_neg_edges = perm[~mask][:num_neg_samples]
    neg_edges = construct_hyper_edges(flatten_neg_edges, arity, n)
    return neg_edges
