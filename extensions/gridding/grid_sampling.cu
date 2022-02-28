/*
 * @Author: Haozhe Xie
 * @Date:   2019-11-13 10:53:22
 * @Last Modified by:   Jerry
 * @Last Modified time: 2022/2/22
 * @not finifshed yet
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

__global__ void grid_sampling_kernel( int n_grid_vertices, 
                                      int n_pts, 
                                      float min_x,
                                      float min_y,
                                      float min_z,
                                      int len_y,
                                      int len_z,
                                      const float *__restrict__ ptcloud,
                                      const float *__restrict__ grid, 
                                      float *__restrict__ grid_pt_weights,
                                      float *__restrict__ grid_pt_indexes,
                                      float *__restrict__ pt_feature){
  // grid : (batch_size, 64, 64, 64, 8)
  // ptcloud : (batch_size, n, 8)
  int batch_index = blockIdx.x;  //一个batch一个block
  int index       = threadIdx.x; //当前线程
  int stride      = blockDim.x;  //每个block的线程数

  ptcloud += batch_index * n_pts * 3;
  grid += batch_index * n_grid_vertices;
  grid_pt_weights += batch_index * n_pts * 24;
  grid_pt_indexes += batch_index * n_pts * 8;

  for(int j = index; j < n_pts; j += stride){
    float pt_x = ptcloud[j * 3 + 0];
    float pt_y = ptcloud[j * 3 + 1];
    float pt_z = ptcloud[j * 3 + 2];

    int lower_x = std::floor(pt_x);
    int upper_x = std::ceil(pt_x);
    if(lower_x == upper_x){
      upper_x += 1;
    }

    int lower_y = std::floor(pt_y);
    int upper_y = std::ceil(pt_y);
    if(lower_y == upper_y){
      upper_y += 1;
    }

    int lower_z = std::floor(pt_z);
    int upper_z = std::ceil(pt_z);
    if(lower_z == upper_z){
      upper_z += 1;
    }

    int lx_offset = lower_x - min_x, ux_offset = upper_x - min_x;
    int ly_offset = lower_y - min_y, uy_offset = upper_y - min_y;
    int lz_offset = lower_z - min_z, uz_offset = upper_z - min_z;

    // 对每个点的8个邻居，计算其相对改点的权重
    // LLL -> Lower X, Lower Y, Lower Z   
    grid_pt_indexes[j * 8 + 0] = 
      compute_index(lx_offset, ly_offset, lz_offset, len_y, len_z);
    grid_pt_weights[j * 24 + 0] = compute_weight(pt_x, lower_x);
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
  int gvtx_feat = 0;
  for (int j = index; j < n_pts; j += stride){
    // LLL
    gvtx_idx = grid_pt_indexes[j * 8 + 0]; //找到对应的网格顶点8个顶点
    // 根据权重计算该点的特征， 8维特征, 受到8个顶点影响
    for(int i = 0; i < 8; i++){//找到对应的8个顶点的特征和权重，每个顶点8个特征
      atomicAdd(&(pt_feature[j+i]), grid[gvtx_idx * 8 + i] * grid_pt_weights[j * 24 + 0]
                                                  * grid_pt_weights[j * 24 + 1]
                                                  * grid_pt_weights[j * 24 + 2]);
    }
    //LLU
    gvtx_idx = grid_pt_indexes[j * 8 + 1];
    for(int i = 0; i < 8; i++){
      atomicAdd(&(pt_feature[j+i]), grid[gvtx_idx * 8 + i] * grid_pt_weights[j * 24 + 3]
                                                  * grid_pt_weights[j * 24 + 4]
                                                  * grid_pt_weights[j * 24 + 5]);
    }

    //LUL
    gvtx_idx = grid_pt_indexes[j * 8 + 2];
    for(int i = 0; i < 8; i++){
      atomicAdd(&(pt_feature[j+i]), grid[gvtx_idx * 8 + i] * grid_pt_weights[j * 24 + 6]
                                                  * grid_pt_weights[j * 24 + 7]
                                                  * grid_pt_weights[j * 24 + 8]); 
    }
    
    //LUU
    gvtx_idx = grid_pt_indexes[j * 8 + 3];
    for(int i = 0; i < 8; i++){
      atomicAdd(&(pt_feature[j+i]), grid[gvtx_idx * 8 + i] * grid_pt_weights[j * 24 + 9]
                                                  * grid_pt_weights[j * 24 + 10]
                                                  * grid_pt_weights[j * 24 + 11]);
    }
      
    //ULL
    gvtx_idx = grid_pt_indexes[j * 8 + 4];
    for(int i = 0; i < 8; i++){
      atomicAdd(&(pt_feature[j+i]), grid[gvtx_idx * 8 + i] * grid_pt_weights[j * 24 + 12]
                                                  * grid_pt_weights[j * 24 + 13]
                                                  * grid_pt_weights[j * 24 + 14]);
    }

    //ULU
    gvtx_idx = grid_pt_indexes[j * 8 + 5];
    for(int i = 0; i < 8; i++){
      atomicAdd(&(pt_feature[j+i]), grid[gvtx_idx * 8 + i] * grid_pt_weights[j * 24 + 15]
                                                  * grid_pt_weights[j * 24 + 16]
                                                  * grid_pt_weights[j * 24 + 17]);
    }

    //UUL
    gvtx_idx = grid_pt_indexes[j * 8 + 6];
    for(int i = 0; i < 8; i++){
      atomicAdd(&(pt_feature[j+i]), grid[gvtx_idx * 8 + i] * grid_pt_weights[j * 24 + 18]
                                                  * grid_pt_weights[j * 24 + 19]
                                                  * grid_pt_weights[j * 24 + 20]);
    }

    //UUU
    gvtx_idx = grid_pt_indexes[j * 8 + 7];
    for(int i = 0; i < 8; i ++){
      atomicAdd(&(pt_feature[j+i]), grid[gvtx_idx * 8 + i] * grid_pt_weights[j * 24 + 21]
                                                  * grid_pt_weights[j * 24 + 22]
                                                  * grid_pt_weights[j * 24 + 23]);
    }
    
  }

}

/* (scale, grid, ptcloud) */
std::vector<torch::Tensor> grid_sampling_cuda_forward(float min_x,
                                                      float min_y,
                                                      float min_z,
                                                      torch::Tensor grid,
                                                      torch::Tensor ptcloud,
                                                      cudaStream_t stream) {
  int batch_size      = ptcloud.size(0);
  int n_pts           = ptcloud.size(1);
  int len_x           = grid.size[0];
  int len_y           = grid.size[1];
  int len_z           = grid.size[2];
  int n_grid_vertices = len_x * len_y * len_z;
  int n_channel       = 8;

  //声明需要计算的变量
  torch::Tensor pt_feature =
    torch::zeros({batch_size, n_pts, n_channel}, torch::CUDA(torch::kFloat));
  torch::Tensor grid_pt_weights = 
    torch::zeros({batch_size, n_pts, 8, 3});
  torch::Tensor grid_pt_indexes = 
    torch::zeros({batch_size, n_pts, 8}, torch::CUDA(torch::kInt));

  gridding_kernel<<<batch_size, get_n_threads(n_pts), 0, stream>>>(
    n_grid_vertices, n_pts, min_x, min_y, min_z, len_y, len_z, 
    ptcloud.data_ptr<float>(), grid.data_ptr<float>(), 
    grid_pt_weights.data_ptr<float>(), grid_pt_indexes.data_ptr<int>()),
    pt_feature.data_ptr<float>()
  );

  cudaError err = cudaGetLastError();
  if(err != cudaSuccess){
    printf("Error in gridding_cuda_forward: %s\n", cudaGetErrorString(err));
  }
  return {pt_feature, grid_pt_weights, grid_pt_indexes};
}

__global__ void grid_sampling_kernel(int n_grid_vertices,
                                     int n_pts,
                                     const float *__restrict__ grad_ptcloud,
                                     const float *__restrict__ grid_pt_weights,
                                     const int *__restrict__ grid_pt_indexes,
                                     float *__restrict__ grad_grid
                                     ) {
  int batch_index = blockIdx.x;  //第i个点云
  int index       = threadIdx.x;  //点云中第j个点
  int stride      = blockDim.x;

  grid_pt_weights += batch_index * n_pts * 24; //每个点到周围网格点的权重影响
  grid_pt_indexes += batch_index * n_pts * 8;  //每个点周围的网格点的索引
  grad_ptcloud += batch_index * n_pts * 3;     //每个点的grad索引
  grad_grid += batch_index * n_grid_vertices;  //网格点存储的grad

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