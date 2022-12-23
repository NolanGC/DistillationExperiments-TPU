#include <torch/extension.h>
#include <algorithm>
#include <vector>
#include <random>

std::vector<uint32_t> random_permutation(uint32_t N, uint32_t index) {
  std::vector<uint32_t> permutation(N);
  std::iota(permutation.begin(), permutation.end(), 0);  // Fill with 0, 1, ..., N-1

  // Swap the number at the specified index with the last element
  std::swap(permutation[index], permutation.back());

  // Shuffle the numbers, excluding the last element (which we want to remain fixed)
  std::shuffle(permutation.begin(), permutation.end() - 1, std::mt19937{std::random_device{}()});

  // Swap the number back to its original position
  std::swap(permutation[index], permutation.back());

  return permutation;
}


at::Tensor batch_permutation_matrix(uint32_t batch_size, uint32_t N, torch::Tensor targets) {
  std::vector<torch::Tensor> tensors;
  for(int64_t i = 0; i < batch_size; ++i){
      std::vector<uint32_t> permuted_indices = random_permutation(N, targets[i].item<int>());
      torch::Tensor tensor = torch::from_blob(permuted_indices.data(), permuted_indices.size(), torch::TensorOptions().dtype(torch::kInt32)).clone();
      auto matrix = torch::zeros({N, N}, tensor.options());
      for (int64_t i = 0; i < N; i++) {
        int64_t elem = tensor[i].item<int64_t>();
        matrix[i][elem] = 1;
      }
      tensors.push_back(matrix);
  }
  return torch::stack(tensors);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("batch_permutation_matrix", &batch_permutation_matrix, "Create a permutation matrix for an N-dimensional tensor with a fixed index");
}
