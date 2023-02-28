#include <vector>
#include <torch/extension.h>
#include <omp.h>
#include <iostream>
#include <set>
#include <cstdlib>
#include <algorithm>
#include <queue>
using std::vector;
using std::set;
using std::swap;
using std::queue;

torch::Tensor cat(
    torch::Tensor rowptr,
    torch::Tensor col,
    int64_t start,
    int64_t end,
    int64_t k,
    int64_t subgraph_size) {
	vector<vector<int64_t>> paths;
	vector<vector<int64_t>> stack;
	stack.push_back({start});
	auto rowptr_data = rowptr.data_ptr<int64_t>();
	auto col_data = col.data_ptr<int64_t>();
	while (!stack.empty()) {
		vector<int64_t> path = stack.back();
		stack.pop_back();
		int64_t current = path.back();
		if (current == end) {
			paths.push_back(path);
			continue;
		}
		if (path.size() > k) 
			continue;
		auto row_start = rowptr_data[current];
		auto row_end = rowptr_data[current + 1];
		for (int64_t i = row_start; i < row_end; i++) {
			int64_t next = col_data[i];
			vector<int64_t> new_path(path);
			new_path.push_back(next);
			stack.push_back(new_path);
		}
	}
	return paths;
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dfs_find_all_path_nodes", &dfs_find_all_path_nodes, "Find all path nodes using dfs");
  m.def("bfs_find_all_path_nodes", &bfs_find_all_path_nodes, "Find all path nodes using bfs");
  m.def("find_all_paths", &find_all_paths, "Find all paths");
}