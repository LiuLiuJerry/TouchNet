/*
 * @Author: Haozhe Xie
 * @Date:   2019-11-13 10:53:22
 * @Last Modified by:   Haozhe Xie
 * @Last Modified time: 2020-06-17 15:00:10
 * @Email:  cshzxie@gmail.com
 */

#include <torch/extension.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#define CUDA_NUM_THREADS 512

// Computer the number of threads needed in GPU
inline int get_n_threads(int n) {
  const int pow_2 = std::log(static_cast<float>(n)) / std::log(2.0);
  return max(min(1 << pow_2, CUDA_NUM_THREADS), 1);
}

__device__ int compute_index(
  int offset_x, int offset_y, int offset_z, int len_y, int len_z) {
  return offset_x * len_y * len_z + offset_y * len_z + offset_z;
}
//根据线性插值计算权重
__device__ float compute_weight(float x, float x0) { return 1 - abs(x - x0); }

__global__ void grid_sampling_kernel(){

}

/* (scale, grid, ptcloud) */
std::vector<torch::Tensor> grid_sampling_cuda_forward(int scale,
                                                 torch::Tensor grid,
                                                 torch::Tensor ptcloud,
                                                 cudaStream_t stream) {

}

__global__ void grid_sampling_kernel(int n_grid_vertices,
                                     int n_pts,
                                     const float *__restrict__ grid_pt_weights,
                                     const int *__restrict__ grid_pt_indexes,
                                     const float *__restrict__ grad_grid,
                                     float *__restrict__ grad_ptcloud) {
  int batch_index = blockIdx.x;  //第i个点云
  int index       = threadIdx.x; //点云中第j个点
  int stride      = blockDim.x;

  grid_pt_weights += batch_index * n_pts * 24; //每个点到周围网格点的权重影响
  grid_pt_indexes += batch_index * n_pts * 8;  //每个点周围的网格点的索引
  grad_grid += batch_index * n_grid_vertices;  //网格点存储的grad
  grad_ptcloud += batch_index * n_pts * 3;     //每个点的grad索引

  int gvtx_idx   = 0;
  float grad_vtx = 0, x_weights = 0, y_weights = 0, z_weights = 0;
  for (int j = index; j < n_pts; j += stride) {
    // Compute gradient for the corresponding positions, a loop for 8 points
    // LLL -> Lower X, Lower Y, Lower Z
    // (x,y,z)每一维度都分别使用重心坐标插值
    gvtx_idx  = grid_pt_indexes[j * 8 + 0];
    grad_vtx  = grad_grid[gvtx_idx]; //点云中第j个点所对应的第1个网格点的梯度
    x_weights = grid_pt_weights[j * 24 + 0]; //该点第1个顶点对应的权重
    y_weights = grid_pt_weights[j * 24 + 1];
    z_weights = grid_pt_weights[j * 24 + 2];
    atomicAdd(&(grad_ptcloud[j * 3 + 0]), -grad_vtx * y_weights * z_weights); //根据点云中的点到网格点的权重将网格点的grad拆分到每个点云上
    atomicAdd(&(grad_ptcloud[j * 3 + 1]), -grad_vtx * x_weights * z_weights);
    atomicAdd(&(grad_ptcloud[j * 3 + 2]), -grad_vtx * x_weights * y_weights);

    // LLU -> Lower X, Lower Y, Upper Z
    gvtx_idx  = grid_pt_indexes[j * 8 + 1];
    grad_vtx  = grad_grid[gvtx_idx];
    x_weights = grid_pt_weights[j * 24 + 3];
    y_weights = grid_pt_weights[j * 24 + 4];
    z_weights = grid_pt_weights[j * 24 + 5];
    atomicAdd(&(grad_ptcloud[j * 3 + 0]), -grad_vtx * y_weights * z_weights);
    atomicAdd(&(grad_ptcloud[j * 3 + 1]), -grad_vtx * x_weights * z_weights);
    atomicAdd(&(grad_ptcloud[j * 3 + 2]), grad_vtx * x_weights * y_weights);

    // LUL -> Lower X, Upper Y, Lower Z
    gvtx_idx  = grid_pt_indexes[j * 8 + 2];
    grad_vtx  = grad_grid[gvtx_idx];
    x_weights = grid_pt_weights[j * 24 + 6];
    y_weights = grid_pt_weights[j * 24 + 7];
    z_weights = grid_pt_weights[j * 24 + 8];
    atomicAdd(&(grad_ptcloud[j * 3 + 0]), -grad_vtx * y_weights * z_weights);
    atomicAdd(&(grad_ptcloud[j * 3 + 1]), grad_vtx * x_weights * z_weights);
    atomicAdd(&(grad_ptcloud[j * 3 + 2]), -grad_vtx * x_weights * y_weights);

    // LUU -> Lower X, Upper Y, Upper Z
    gvtx_idx  = grid_pt_indexes[j * 8 + 3];
    grad_vtx  = grad_grid[gvtx_idx];
    x_weights = grid_pt_weights[j * 24 + 9];
    y_weights = grid_pt_weights[j * 24 + 10];
    z_weights = grid_pt_weights[j * 24 + 11];
    atomicAdd(&(grad_ptcloud[j * 3 + 0]), -grad_vtx * y_weights * z_weights);
    atomicAdd(&(grad_ptcloud[j * 3 + 1]), grad_vtx * x_weights * z_weights);
    atomicAdd(&(grad_ptcloud[j * 3 + 2]), grad_vtx * x_weights * y_weights);

    // ULL -> Upper X, Lower Y, Lower Z
    gvtx_idx  = grid_pt_indexes[j * 8 + 4];
    grad_vtx  = grad_grid[gvtx_idx];
    x_weights = grid_pt_weights[j * 24 + 12];
    y_weights = grid_pt_weights[j * 24 + 13];
    z_weights = grid_pt_weights[j * 24 + 14];
    atomicAdd(&(grad_ptcloud[j * 3 + 0]), grad_vtx * y_weights * z_weights);
    atomicAdd(&(grad_ptcloud[j * 3 + 1]), -grad_vtx * x_weights * z_weights);
    atomicAdd(&(grad_ptcloud[j * 3 + 2]), -grad_vtx * x_weights * y_weights);

    // ULU -> Upper X, Lower Y, Upper Z
    gvtx_idx  = grid_pt_indexes[j * 8 + 5];
    grad_vtx  = grad_grid[gvtx_idx];
    x_weights = grid_pt_weights[j * 24 + 15];
    y_weights = grid_pt_weights[j * 24 + 16];
    z_weights = grid_pt_weights[j * 24 + 17];
    atomicAdd(&(grad_ptcloud[j * 3 + 0]), grad_vtx * y_weights * z_weights);
    atomicAdd(&(grad_ptcloud[j * 3 + 1]), -grad_vtx * x_weights * z_weights);
    atomicAdd(&(grad_ptcloud[j * 3 + 2]), grad_vtx * x_weights * y_weights);

    // UUL -> Upper X, Upper Y, Lower Z
    gvtx_idx  = grid_pt_indexes[j * 8 + 6];
    grad_vtx  = grad_grid[gvtx_idx];
    x_weights = grid_pt_weights[j * 24 + 18];
    y_weights = grid_pt_weights[j * 24 + 19];
    z_weights = grid_pt_weights[j * 24 + 20];
    atomicAdd(&(grad_ptcloud[j * 3 + 0]), grad_vtx * y_weights * z_weights);
    atomicAdd(&(grad_ptcloud[j * 3 + 1]), grad_vtx * x_weights * z_weights);
    atomicAdd(&(grad_ptcloud[j * 3 + 2]), -grad_vtx * x_weights * y_weights);

    // UUU -> Upper X, Upper Y, Upper Z
    gvtx_idx  = grid_pt_indexes[j * 8 + 7];
    grad_vtx  = grad_grid[gvtx_idx];
    x_weights = grid_pt_weights[j * 24 + 21];
    y_weights = grid_pt_weights[j * 24 + 22];
    z_weights = grid_pt_weights[j * 24 + 23];
    atomicAdd(&(grad_ptcloud[j * 3 + 0]), grad_vtx * y_weights * z_weights);
    atomicAdd(&(grad_ptcloud[j * 3 + 1]), grad_vtx * x_weights * z_weights);
    atomicAdd(&(grad_ptcloud[j * 3 + 2]), grad_vtx * x_weights * y_weights);
  }
}

torch::Tensor grid_sampling_cuda_backward(torch::Tensor grid_pt_weights,
                                     torch::Tensor grid_pt_indexes,
                                     torch::Tensor grad_grid,
                                     cudaStream_t stream) {
  int batch_size      = grad_grid.size(0);
  int n_grid_vertices = grad_grid.size(1);
  int n_pts           = grid_pt_indexes.size(1);

  torch::Tensor grad_ptcloud =
    torch::zeros({batch_size, n_pts, 3}, torch::CUDA(torch::kFloat));

  gridding_grad_kernel<<<batch_size, get_n_threads(n_pts), 0, stream>>>(
    n_grid_vertices, n_pts, grid_pt_weights.data_ptr<float>(),
    grid_pt_indexes.data_ptr<int>(), grad_grid.data_ptr<float>(),
    grad_ptcloud.data_ptr<float>());

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Error in gridding_cuda_backward: %s\n", cudaGetErrorString(err));
  }
  return grad_ptcloud;
}