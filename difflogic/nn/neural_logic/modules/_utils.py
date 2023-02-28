#! /usr/bin/env python3
#

"""Utility functions for tensor masking."""

import torch

import torch.autograd as ag
from jactorch.functional import meshgrid, meshgrid_exclude_self
import matplotlib.pyplot as plt
import networkx as nx

__all__ = ['meshgrid', 'meshgrid_exclude_self', 'exclude_mask', 'mask_value']

color_map = ['r', 'g', 'b', 'm', 'c', 'y', 'k', 'purple']


def exclude_mask(inputs, cnt=2, dim=1):
    """Produce an exclusive mask.

    Specifically, for cnt=2, given an array a[i, j] of n * n, it produces
    a mask with size n * n where only a[i, j] = 1 if and only if (i != j).

    Args:
      inputs: The tensor to be masked.
      cnt: The operation is performed over [dim, dim + cnt) axes.
      dim: The starting dimension for the exclusive mask.

    Returns:
      A mask that make sure the coordinates are mutually exclusive.
    """
    assert cnt > 0
    if dim < 0:
        dim += inputs.dim()
    n = inputs.size(dim)
    for i in range(1, cnt):
        assert n == inputs.size(dim + i)

    rng = torch.arange(0, n, dtype=torch.long, device=inputs.device)
    q = []
    for i in range(cnt):
        p = rng
        for j in range(cnt):
            if i != j:
                p = p.unsqueeze(j)
        p = p.expand((n,) * cnt)
        q.append(p)
    mask = q[0] == q[0]
    # Mutually Exclusive
    for i in range(cnt):
        for j in range(cnt):
            if i != j:
                mask *= q[i] != q[j]
    for i in range(dim):
        mask.unsqueeze_(0)
    for j in range(inputs.dim() - dim - cnt):
        mask.unsqueeze_(-1)
    return mask.expand(inputs.size()).float()


def mask_value(inputs, mask, value):
    assert inputs.size() == mask.size()
    return inputs * mask + value * (1 - mask)


def sparse_exclude_mask(inputs, cnt=2, dim=1):
    """Produce an exclusive mask.

    Specifically, for cnt=2, given an array a[i, j] of n * n, it produces
    a mask with size n * n where only a[i, j] = 1 if and only if (i != j).

    Args:
      inputs: The tensor to be masked.
      cnt: The operation is performed over [dim, dim + cnt) axes.
      dim: The starting dimension for the exclusive mask.

    Returns:
      A mask that make sure the coordinates are mutually exclusive.
    """
    if dim < 0:
        dim += inputs.dim
    n = inputs.shape[0]

    rng = torch.arange(0, n, dtype=torch.long, device=inputs.device)
    q = []
    for i in range(cnt):
        p = rng
        for j in range(cnt):
            if i != j:
                p = p.unsqueeze(j)
        p = p.expand((n,) * cnt)
        q.append(p)
    mask = q[0] == q[0]
    # Mutually Exclusive
    for i in range(cnt):
        for j in range(cnt):
            if i != j:
                mask *= q[i] != q[j]
    for i in range(dim):
        mask.unsqueeze_(0)
    for j in range(inputs.dim - dim - cnt):
        mask.unsqueeze_(-1)
    return mask.expand(inputs.size()).float()


def sparse_mask_value(inputs, mask, value):
    return inputs * mask + value * (1 - mask)


def plot(inputs, prefix):
    """Plot a graph."""
    t = inputs[2][0]
    filename = prefix+f"-{2}.png"
    channel = t.channel
    coordinate = t.coordinate()
    fig = plt.figure(figsize=(12, 12))
    ax = plt.subplot(111)
    ax.set_title(filename, fontsize=24)
    G = nx.MultiDiGraph()
    G.add_nodes_from(list(range(t.n)))
    for r in range(channel):
        mask = t.val[:, r] > 0.1
        num = mask.sum()
        edges = coordinate[mask].cpu().tolist()
        prob = t.val[mask, r].cpu().tolist()
        edges = [(*l, {'rel': r, 'prob': p}) for l, p in zip(edges, prob)]
        G.add_edges_from(edges)
    color = [color_map[rel]
             for rel in list(nx.get_edge_attributes(G, 'rel').values())]
    width = [
        5 * prob for prob in list(nx.get_edge_attributes(G, 'prob').values())]
    pos = nx.shell_layout(G)
    nx.draw(G, pos, node_size=500, with_labels=True, node_color='yellow',
            font_size=24, font_weight='bold', connectionstyle='arc3, rad = 0.05', edge_color=color, width=width, arrowsize=30)
    plt.tight_layout()
    plt.savefig(filename, format="PNG")


def print_groundings(inp):
    if torch.is_tensor(inp):
        ndim = inp.ndim
        print(inp.permute((0, ndim-1, *list(range(1, ndim-1)))))
    else:
        print(inp)