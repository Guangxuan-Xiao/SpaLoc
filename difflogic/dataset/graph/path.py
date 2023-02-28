from ops import dfs_find_all_path_nodes, bfs_find_all_path_nodes


class AdjacencyList:
    def __init__(self, adj_t):
        self.rowptr, self.col, _ = adj_t.csr()
        self.n = self.rowptr.shape[0] - 1

    def __getitem__(self, node):
        neighbors = self.col[self.rowptr[node]:self.rowptr[node + 1]]
        return neighbors


def find_all_path_nodes_py(graph: AdjacencyList, start, end, k, subgraph_size):
    nodes = {start, end}
    stack = [[start]]
    while stack:
        path = stack.pop()
        current = path[-1]
        if current == end:
            # print(path)
            for node in path[1:-1]:
                nodes.add(node)
                if len(nodes) >= subgraph_size:
                    return nodes
            continue
        length = len(path)
        if length > k:
            continue
        neighbors = graph[current]
        # neighbors = neighbors[torch.randperm(len(neighbors))]
        for neighbor in neighbors:
            stack.append(path + [neighbor])
    return nodes


def find_all_path_nodes_c(graph: AdjacencyList, start, end, k, subgraph_size) -> set:
    return dfs_find_all_path_nodes(graph.rowptr, graph.col, start, end, k, subgraph_size)
