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
<<<<<<< HEAD
//根据线性插值计算权重
=======

>>>>>>> d797c9a3b87a78c76f852e04ce9809d4580e0d71
__device__ float compute_weight(float x, float x0) { return 1 - abs(x - x0); }

__global__ void gridding_kernel(int n_grid_vertices,
                                int n_pts,
                                float min_x,
                                float min_y,
                                float min_z,
                                int len_y,
                                int len_z,
                                const float *__restrict__ ptcloud,
                                float *__restrict__ grid_weights,
                                float *__restrict__ grid_pt_weights,
                                int *__restrict__ grid_pt_indexes) {
  int batch_index = blockIdx.x;
  int index       = threadIdx.x;
  int stride      = blockDim.x; //每个block的线程数

  ptcloud += batch_index * n_pts * 3; //指向当前点云的指针
  grid_weights += batch_index * n_grid_vertices; //指向当前网格点的指针
  grid_pt_weights += batch_index * n_pts * 24; //每个点到周围8个网格点产生的权重
  grid_pt_indexes += batch_index * n_pts * 8; //每个点的周围8个网格点的索引

  for (int j = index; j < n_pts; j += stride) { //如果一个点云的点的数目一个block放不下的话，则进入下一个循环
    float pt_x = ptcloud[j * 3 + 0];
    float pt_y = ptcloud[j * 3 + 1];
    float pt_z = ptcloud[j * 3 + 2];

<<<<<<< HEAD
    int lower_x = std::floor(pt_x); //此处应该已经除以网格大小了，因此直接取整数即可得到其在第几个网格中
=======
    int lower_x = std::floor(pt_x); //此处应该已经处以网格大小了，因此直接取整数即可得到其在第几个网格中
>>>>>>> d797c9a3b87a78c76f852e04ce9809d4580e0d71
    int upper_x = std::ceil(pt_x);
    if (lower_x == upper_x) {
      upper_x += 1;
    }
    int lower_y = std::floor(pt_y);
    int upper_y = std::ceil(pt_y);
    if (lower_y == upper_y) {
      upper_y += 1;
    }
    int lower_z = std::floor(pt_z);
    int upper_z = std::ceil(pt_z);
    if (lower_z == upper_z) {
      upper_z += 1;
    }

    int lx_offset = lower_x - min_x, ux_offset = upper_x - min_x;
    int ly_offset = lower_y - min_y, uy_offset = upper_y - min_y;
    int lz_offset = lower_z - min_z, uz_offset = upper_z - min_z;

    // Compute weights and corresponding positions, a loop for 8 points
    // LLL -> Lower X, Lower Y, Lower Z
    grid_pt_indexes[j * 8 + 0] =
      compute_index(lx_offset, ly_offset, lz_offset, len_y, len_z); //计算每个点对应的网格点的索引
    grid_pt_weights[j * 24 + 0] = compute_weight(pt_x, lower_x); //计算每个点给周围网格顶点的权重影响，一共8个网格顶点
    grid_pt_weights[j * 24 + 1] = compute_weight(pt_y, lower_y);
    grid_pt_weights[j * 24 + 2] = compute_weight(pt_z, lower_z);

    // LLU -> Lower X, Lower Y, Upper Z
    grid_pt_indexes[j * 8 + 1] =
      compute_index(lx_offset, ly_offset, uz_offset, len_y, len_z);
    grid_pt_weights[j * 24 + 3] = compute_weight(pt_x, lower_x);
    grid_pt_weights[j * 24 + 4] = compute_weight(pt_y, lower_y);
    grid_pt_weights[j * 24 + 5] = compute_weight(pt_z, upper_z);

    // LUL -> Lower X, Upper Y, Lower Z
    grid_pt_indexes[j * 8 + 2] =
      compute_index(lx_offset, uy_offset, lz_offset, len_y, len_z);
    grid_pt_weights[j * 24 + 6] = compute_weight(pt_x, lower_x);
    grid_pt_weights[j * 24 + 7] = compute_weight(pt_y, upper_y);
    grid_pt_weights[j * 24 + 8] = compute_weight(pt_z, lower_z);

    // LUU -> Lower X, Upper Y, Upper Z
    grid_pt_indexes[j * 8 + 3] =
      compute_index(lx_offset, uy_offset, uz_offset, len_y, len_z);
    grid_pt_weights[j * 24 + 9]  = compute_weight(pt_x, lower_x);
    grid_pt_weights[j * 24 + 10] = compute_weight(pt_y, upper_y);
    grid_pt_weights[j * 24 + 11] = compute_weight(pt_z, upper_z);

    // ULL -> Upper X, Lower Y, Lower Z
    grid_pt_indexes[j * 8 + 4] =
      compute_index(ux_offset, ly_offset, lz_offset, len_y, len_z);
    grid_pt_weights[j * 24 + 12] = compute_weight(pt_x, upper_x);
    grid_pt_weights[j * 24 + 13] = compute_weight(pt_y, lower_y);
    grid_pt_weights[j * 24 + 14] = compute_weight(pt_z, lower_z);

    // ULU -> Upper X, Lower Y, Upper Z
    grid_pt_indexes[j * 8 + 5] =
      compute_index(ux_offset, ly_offset, uz_offset, len_y, len_z);
    grid_pt_weights[j * 24 + 15] = compute_weight(pt_x, upper_x);
    grid_pt_weights[j * 24 + 16] = compute_weight(pt_y, lower_y);
    grid_pt_weights[j * 24 + 17] = compute_weight(pt_z, upper_z);

    // UUL -> Upper X, Upper Y, Lower Z
    grid_pt_indexes[j * 8 + 6] =
      compute_index(ux_offset, uy_offset, lz_offset, len_y, len_z);
    grid_pt_weights[j * 24 + 18] = compute_weight(pt_x, upper_x);
    grid_pt_weights[j * 24 + 19] = compute_weight(pt_y, upper_y);
    grid_pt_weights[j * 24 + 20] = compute_weight(pt_z, lower_z);

    // UUU -> Upper X, Upper Y, Upper Z
    grid_pt_indexes[j * 8 + 7] =
      compute_index(ux_offset, uy_offset, uz_offset, len_y, len_z);
    grid_pt_weights[j * 24 + 21] = compute_weight(pt_x, upper_x);
    grid_pt_weights[j * 24 + 22] = compute_weight(pt_y, upper_y);
    grid_pt_weights[j * 24 + 23] = compute_weight(pt_z, upper_z);
  }

  __syncthreads();

  int gvtx_idx = 0;
  for (int j = index; j < n_pts; j += stride) {
    // LLL -> Lower X, Lower Y, Lower Z
<<<<<<< HEAD
    // 定义了一种特征传递方法，即将顶点对周围的每个网格的x y z轴的权重乘起来，加到网格对应的权重
    // 存储数组grid_weights[gvtx_idx]
    gvtx_idx = grid_pt_indexes[j * 8 + 0]; 
=======
    gvtx_idx = grid_pt_indexes[j * 8 + 0]; //将每个网格顶点x y z轴的权重乘起来， 加到网格对应的权重存储grid_weights上
>>>>>>> d797c9a3b87a78c76f852e04ce9809d4580e0d71
    atomicAdd(&(grid_weights[gvtx_idx]), grid_pt_weights[j * 24 + 0] *
                                           grid_pt_weights[j * 24 + 1] *
                                           grid_pt_weights[j * 24 + 2]);
    // LLU -> Lower X, Lower Y, Upper Z
    gvtx_idx = grid_pt_indexes[j * 8 + 1];
    atomicAdd(&(grid_weights[gvtx_idx]), grid_pt_weights[j * 24 + 3] *
                                           grid_pt_weights[j * 24 + 4] *
                                           grid_pt_weights[j * 24 + 5]);
    // LUL -> Lower X, Upper Y, Lower Z
    gvtx_idx = grid_pt_indexes[j * 8 + 2];
    atomicAdd(&(grid_weights[gvtx_idx]), grid_pt_weights[j * 24 + 6] *
                                           grid_pt_weights[j * 24 + 7] *
                                           grid_pt_weights[j * 24 + 8]);
    // LUU -> Lower X, Upper Y, Upper Z
    gvtx_idx = grid_pt_indexes[j * 8 + 3];
    atomicAdd(&(grid_weights[gvtx_idx]), grid_pt_weights[j * 24 + 9] *
                                           grid_pt_weights[j * 24 + 10] *
                                           grid_pt_weights[j * 24 + 11]);
    // ULL -> Upper X, Lower Y, Lower Z
    gvtx_idx = grid_pt_indexes[j * 8 + 4];
    atomicAdd(&(grid_weights[gvtx_idx]), grid_pt_weights[j * 24 + 12] *
                                           grid_pt_weights[j * 24 + 13] *
                                           grid_pt_weights[j * 24 + 14]);
    // ULU -> Upper X, Lower Y, Upper Z
    gvtx_idx = grid_pt_indexes[j * 8 + 5];
    atomicAdd(&(grid_weights[gvtx_idx]), grid_pt_weights[j * 24 + 15] *
                                           grid_pt_weights[j * 24 + 16] *
                                           grid_pt_weights[j * 24 + 17]);
    // UUL -> Upper X, Upper Y, Lower Z
    gvtx_idx = grid_pt_indexes[j * 8 + 6];
    atomicAdd(&(grid_weights[gvtx_idx]), grid_pt_weights[j * 24 + 18] *
                                           grid_pt_weights[j * 24 + 19] *
                                           grid_pt_weights[j * 24 + 20]);
    // UUU -> Upper X, Upper Y, Upper Z
    gvtx_idx = grid_pt_indexes[j * 8 + 7];
    atomicAdd(&(grid_weights[gvtx_idx]), grid_pt_weights[j * 24 + 21] *
                                           grid_pt_weights[j * 24 + 22] *
                                           grid_pt_weights[j * 24 + 23]);
  }
}

std::vector<torch::Tensor> gridding_cuda_forward(float min_x,
                                                 float max_x,
                                                 float min_y,
                                                 float max_y,
                                                 float min_z,
                                                 float max_z,
                                                 torch::Tensor ptcloud,
                                                 cudaStream_t stream) {
  int batch_size      = ptcloud.size(0);
  int n_pts           = ptcloud.size(1);
  int len_x           = max_x - min_x + 1;
  int len_y           = max_y - min_y + 1;
  int len_z           = max_z - min_z + 1;
  int n_grid_vertices = len_x * len_y * len_z;
<<<<<<< HEAD
  //声明变量
=======

>>>>>>> d797c9a3b87a78c76f852e04ce9809d4580e0d71
  torch::Tensor grid_weights =
    torch::zeros({batch_size, n_grid_vertices}, torch::CUDA(torch::kFloat));
  torch::Tensor grid_pt_weights =
    torch::zeros({batch_size, n_pts, 8, 3}, torch::CUDA(torch::kFloat));
  torch::Tensor grid_pt_indexes =
    torch::zeros({batch_size, n_pts, 8}, torch::CUDA(torch::kInt));

  gridding_kernel<<<batch_size, get_n_threads(n_pts), 0, stream>>>(
    n_grid_vertices, n_pts, min_x, min_y, min_z, len_y, len_z,
    ptcloud.data_ptr<float>(), grid_weights.data_ptr<float>(),
    grid_pt_weights.data_ptr<float>(), grid_pt_indexes.data_ptr<int>());

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Error in gridding_cuda_forward: %s\n", cudaGetErrorString(err));
  }
  return {grid_weights, grid_pt_weights, grid_pt_indexes};
}

__global__ void gridding_grad_kernel(int n_grid_vertices,
                                     int n_pts,
                                     const float *__restrict__ grid_pt_weights,
                                     const int *__restrict__ grid_pt_indexes,
                                     const float *__restrict__ grad_grid,
                                     float *__restrict__ grad_ptcloud) {
  int batch_index = blockIdx.x; //第i个点云
  int index       = threadIdx.x; //点云中第j个点
  int stride      = blockDim.x;

  grid_pt_weights += batch_index * n_pts * 24;  //每个点到周围网格点的权重影响
  grid_pt_indexes += batch_index * n_pts * 8;  //每个点周围的网格点的索引
  grad_grid += batch_index * n_grid_vertices;  //网格点存储的grad
  grad_ptcloud += batch_index * n_pts * 3; //每个点的grad索引

  int gvtx_idx   = 0;
  float grad_vtx = 0, x_weights = 0, y_weights = 0, z_weights = 0;
  for (int j = index; j < n_pts; j += stride) {
    // Compute gradient for the corresponding positions, a loop for 8 points
    // LLL -> Lower X, Lower Y, Lower Z
<<<<<<< HEAD
    // (x,y,z)每一维度都分别使用重心坐标插值
    gvtx_idx  = grid_pt_indexes[j * 8 + 0];
    grad_vtx  = grad_grid[gvtx_idx]; //点云中第j个点所对应的第1个网格点的梯度
    x_weights = grid_pt_weights[j * 24 + 0]; //该点第1个顶点对应的权重
    y_weights = grid_pt_weights[j * 24 + 1];
    z_weights = grid_pt_weights[j * 24 + 2];
    //链式法则，后面一层的梯度为grad_vtx，这里相当将[1-(p_x-low_x)]对p_x求导，得到-1
    //维度上，从1维拆解为3维
=======
    gvtx_idx  = grid_pt_indexes[j * 8 + 0];
    grad_vtx  = grad_grid[gvtx_idx];
    x_weights = grid_pt_weights[j * 24 + 0];
    y_weights = grid_pt_weights[j * 24 + 1];
    z_weights = grid_pt_weights[j * 24 + 2];
>>>>>>> d797c9a3b87a78c76f852e04ce9809d4580e0d71
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

torch::Tensor gridding_cuda_backward(torch::Tensor grid_pt_weights,
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