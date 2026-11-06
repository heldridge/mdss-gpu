#pragma once

#include <stdexcept>

#include "math/ff_math.h"

template <typename Field>
__global__ void compute_barycentric_weights_device(const uint32_t* d_x_coords,
                                                   uint32_t* d_weights,
                                                   int num_points) {
  int i = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (i < num_points) {
    uint32_t xi = d_x_coords[i];
    uint32_t weight = 1;

    for (int j = 0; j < num_points; ++j) {
      if (i != j) {
        weight = Field::mul(weight, Field::sub(xi, d_x_coords[j]));
      }
    }

    d_weights[i] = Field::inverse(weight);
  }
}

template <typename Field>
void FFMath<Field>::compute_barycentric_weights(const uint32_t* d_x_coords,
                                                uint32_t* d_weights,
                                                int num_points) {
  int blocks_n = (num_points + 255) / 256;
  compute_barycentric_weights_device<Field>
      <<<blocks_n, 256>>>(d_x_coords, d_weights, num_points);
  cudaDeviceSynchronize();
}

template <typename Field>
__global__ void compute_update_M_step(const uint32_t* d_M_curr,
                                      uint32_t* d_M_next,
                                      const uint32_t* d_x_coords, int i) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j <= i + 1) {
    uint32_t xi = d_x_coords[i];
    uint32_t left = (j > 0) ? d_M_curr[j - 1] : 0;
    uint32_t right = (j <= i) ? d_M_curr[j] : 0;

    // M_next_j = M_curr_{j-1} - xi * M_curr_j
    d_M_next[j] = Field::sub(left, Field::mul(xi, right));
  }
}

template <typename Field>
void FFMath<Field>::compute_master_polynomial(const uint32_t* d_x_coords,
                                              uint32_t*& d_M, int num_points) {
  unsigned long size = sizeof(uint32_t);
  uint32_t* d_M_next;
  cudaMalloc(&d_M_next, size * (num_points + 1));

  cudaMemset(d_M, 0, size * (num_points + 1));
  uint32_t one[1] = {1};
  cudaMemcpy(d_M, &one, size, cudaMemcpyHostToDevice);

  for (int i = 0; i < num_points; ++i) {
    int blocks_M = (i + 2 + 255) / 256;
    compute_update_M_step<Field>
        <<<blocks_M, 256>>>(d_M, d_M_next, d_x_coords, i);
    std::swap(d_M, d_M_next);
  }

  cudaFree(d_M_next);
}

template <typename Field>
__global__ void init_b_curr_device(uint32_t* d_b_curr, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    d_b_curr[i] = 1;
  }
}

template <typename Field>
void FFMath<Field>::init_b_curr(uint32_t* d_b_curr, int num_points) {
  int blocks_n = (num_points + 255) / 256;
  init_b_curr_device<Field><<<blocks_n, 256>>>(d_b_curr, num_points);
}

template <typename Field>
__global__ void compute_b_chunk_device(uint32_t* d_b_curr, const uint32_t* d_M,
                                       const uint32_t* d_x_coords,
                                       uint32_t* d_B_chunk, int j_start, int S,
                                       int num_points) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num_points) {
    uint32_t xi = d_x_coords[i];
    uint32_t b = d_b_curr[i];

    for (int s = 0; s < S; ++s) {
      int j = j_start - s;
      if (j < 0) break;

      d_B_chunk[s * num_points + i] = b;
      b = Field::add(d_M[j], Field::mul(xi, b));
    }

    d_b_curr[i] = b;
  }
}
template <typename Field>
void FFMath<Field>::compute_b_chunk(uint32_t* d_b_curr, const uint32_t* d_M,
                                    const uint32_t* d_x_coords,
                                    uint32_t* d_B_chunk, int j_start, int S,
                                    int num_points) {
  int blocks_n = (num_points + 255) / 256;
  compute_b_chunk_device<Field><<<blocks_n, 256>>>(
      d_b_curr, d_M, d_x_coords, d_B_chunk, j_start, S, num_points);
}

template <typename Field>
__global__ void compute_C_chunk_device(const uint32_t* y_vals,
                                       const uint32_t* weights,
                                       const uint32_t* B_chunk, uint32_t* C_out,
                                       int j_start, int S, int num_points,
                                       int c) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;  // Polynomial index
  int s = blockIdx.y;                             // Chunk offset index

  if (k < c && s < S) {
    int j = j_start - s;
    if (j < 0) return;

    uint32_t sum = 0;

    for (int i = 0; i < num_points; ++i) {
      uint32_t v_val = Field::mul(y_vals[(i * c) + k], weights[i]);
      uint32_t b_val = B_chunk[(s * num_points) + i];
      sum = Field::add(sum, Field::mul(v_val, b_val));
    }

    C_out[(k * num_points) + j] = sum;
  }
}

template <typename Field>
void FFMath<Field>::compute_c_chunk(const uint32_t* d_y_vals,
                                    const uint32_t* d_weights,
                                    const uint32_t* d_B_chunk,
                                    uint32_t* d_C_out, int j_start, int S,
                                    int num_points, int c) {
  dim3 gridC((c + 255) / 256, S);
  dim3 blockC(256);

  compute_C_chunk_device<Field><<<gridC, blockC>>>(
      d_y_vals, d_weights, d_B_chunk, d_C_out, j_start, S, num_points, c);
}

template <typename Field>
void FFMath<Field>::compute_lagrange_coefficients(
    const uint32_t* d_x_coords, const uint32_t* d_y_coords, const uint32_t* d_M,
    uint32_t* d_lagrange_coeffs, int num_points, int c, int chunk_size) {
  /*
   * d_y_coords should be arranged in row major form. E.g. if your points
   * are (15, (3, 1)), (8, (9, 0)), (14, (14, 6)), then it should be
   * [3, 1, 9, 0, 14, 6]
   */

  if (chunk_size < 1) {
    throw std::invalid_argument("Chunk size must be greater than zero");
  }

  unsigned long size = sizeof(uint32_t);

  uint32_t *d_weights, *d_b_curr, *d_B_chunk;
  cudaMalloc(&d_weights, size * num_points);
  cudaMalloc(&d_b_curr, size * num_points);
  cudaMalloc(&d_B_chunk, size * num_points * chunk_size);

  compute_barycentric_weights(d_x_coords, d_weights, num_points);

  init_b_curr(d_b_curr, num_points);
  for (int j_start = num_points - 1; j_start >= 0; j_start -= chunk_size) {
    compute_b_chunk(d_b_curr, d_M, d_x_coords, d_B_chunk, j_start, chunk_size,
                    num_points);

    compute_c_chunk(d_y_coords, d_weights, d_B_chunk, d_lagrange_coeffs,
                    j_start, chunk_size, num_points, c);
  }

  cudaFree(d_weights);
  cudaFree(d_b_curr);
  cudaFree(d_B_chunk);
}
