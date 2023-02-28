#include <torch/extension.h>
#include <vector>
#include <tuple>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
subgraph(torch::Tensor idx, torch::Tensor rowptr, torch::Tensor row,
             torch::Tensor col)
{
    // assoc is a map from orignal index to subgraph index
    auto assoc = torch::full({rowptr.size(0) - 1}, -1, idx.options());
    assoc.index_copy_(0, idx, torch::arange(idx.size(0), idx.options()));

    auto idx_data = idx.data_ptr<int64_t>();
    auto rowptr_data = rowptr.data_ptr<int64_t>();
    auto col_data = col.data_ptr<int64_t>();
    auto assoc_data = assoc.data_ptr<int64_t>();

    std::vector<int64_t> rows, cols, indices;

    int64_t v, w, w_new, row_start, row_end;
    for (int64_t v_new = 0; v_new < idx.size(0); v_new++)
    {
        v = idx_data[v_new];
        row_start = rowptr_data[v];
        row_end = rowptr_data[v + 1];

        for (int64_t j = row_start; j < row_end; j++)
        {
            w = col_data[j];
            w_new = assoc_data[w];
            if (w_new > -1)
            {
                rows.push_back(v_new);
                cols.push_back(w_new);
                indices.push_back(j);
            }
        }
    }

    int64_t length = rows.size();
    row = torch::from_blob(rows.data(), {length}, row.options()).clone();
    col = torch::from_blob(cols.data(), {length}, row.options()).clone();
    idx = torch::from_blob(indices.data(), {length}, row.options()).clone();

    return std::make_tuple(row, col, idx);
}

std::tuple<torch::Tensor, torch::Tensor>
sub_hypergraph(torch::Tensor node_idx, torch::Tensor edge_index, int64_t num_nodes)
{
    // the shape of edge_index is [num_edges, arity]
    auto assoc = torch::full({num_nodes}, -1, node_idx.options());
    assoc.index_copy_(0, node_idx, torch::arange(node_idx.size(0), node_idx.options()));

    auto edge_index_acc = edge_index.accessor<int64_t, 2>();
    int64_t num_edges = edge_index.size(0), arity = edge_index.size(1);
    std::vector<int64_t> new_edge_index, indices;

    for (int64_t i = 0; i < num_edges; i++)
    {
        bool is_in_subgraph = true;
        for (int64_t j = 0; j < arity; j++)
        {
            if (assoc[edge_index_acc[i][j]] == -1)
            {
                is_in_subgraph = false;
                break;
            }
        }
        if (is_in_subgraph)
        {
            
            for (int64_t j = 0; j < arity; j++)
            {
                new_edge_index.push_back(assoc[edge_index_acc[i][j]]);
            }
            indices.push_back(i);
        }
    }
}