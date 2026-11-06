#include "math/ff_math.h"

std::ostream& operator<<(std::ostream& os, const EROConfig& ec) {
  return os << "{" << ec.target_row_idx << ", " << ec.source_row_idx << ", "
            << ec.zeroing_coefficient << ", " << ec.shift << "}";
}

std::ostream& operator<<(std::ostream& os, const gpuPivotInfo& gpi) {
  return os << "{" << gpi.row_idx << ", " << gpi.degree << "}";
}

WPFData init_wpf_data_structures(uint32_t* d_arr, ArrayMetadata d_arr_meta) {
  int size = sizeof(uint32_t);

  gpuPivotInfo* d_rows_with_pivot;
  int *d_conflicts_stack, *d_conflicts_stack_top, *d_rwp_tops,
      *d_col_best_degree;

  cudaMalloc(&d_conflicts_stack, d_arr_meta.num_cols * size);
  cudaMalloc(&d_conflicts_stack_top, sizeof(int));
  cudaMalloc(&d_rows_with_pivot,
             d_arr_meta.num_cols * d_arr_meta.num_rows * sizeof(gpuPivotInfo));
  cudaMalloc(&d_rwp_tops, d_arr_meta.num_cols * sizeof(int));
  cudaMalloc(&d_col_best_degree, d_arr_meta.num_cols * sizeof(int));

  cudaMemset(d_conflicts_stack_top, 0, sizeof(int));
  cudaMemset(d_rwp_tops, 0, d_arr_meta.num_cols * sizeof(int));

  return {d_arr,
          d_arr_meta,
          d_rows_with_pivot,
          d_rwp_tops,
          d_conflicts_stack,
          d_conflicts_stack_top,
          d_col_best_degree};
}

__global__ void init_metadata_for_row(WPFData d, int row_idx) {
  int best_pivot_col = -1;
  int best_degree = -1;

  for (int current_col = 0; current_col < d.d_arr_meta.num_cols;
       current_col++) {
    int current_degree = -1;

    for (int z_idx = d.d_arr_meta.max_degree - 1; z_idx >= 0; z_idx--) {
      if (d.d_arr[(row_idx * d.d_arr_meta.num_cols * d.d_arr_meta.max_degree) +
                  (current_col * d.d_arr_meta.max_degree) + z_idx] > 0) {
        current_degree = z_idx;
        break;
      }
    }

    if (current_degree >= best_degree) {
      best_degree = current_degree;
      best_pivot_col = current_col;
    }
  }

  if (best_degree >= 0) {
    push_stack(d.d_rows_with_pivot + (best_pivot_col * d.d_arr_meta.num_rows),
               d.d_rwp_tops[best_pivot_col], {row_idx, best_degree});
    if (d.d_rwp_tops[best_pivot_col] > 1) {
      push_stack(d.d_conflicts_stack, *d.d_conflicts_stack_top, best_pivot_col);
    }
  }
}

void init_metadata(WPFData d) {
  for (int row_idx = 0; row_idx < d.d_arr_meta.num_rows; ++row_idx) {
    init_metadata_for_row<<<1, 1>>>(d, row_idx);
    cudaDeviceSynchronize();
  }
}

__global__ void _update_metadata_post_ero(WPFData d, EROConfig* cERO) {
  EROConfig ero = cERO[0];
  if (ero.shift < 0) return;

  int target_row = ero.target_row_idx;
  uint32_t col = blockIdx.x;
  if (col >= d.d_arr_meta.num_cols) return;

  extern __shared__ int s_warp_best[];

  int tid = threadIdx.x;
  int lane = tid % 32;
  int warp_id = tid / 32;
  int num_warps = (blockDim.x + 31) / 32;

  uint32_t* col_data =
      d.d_arr + (target_row * d.d_arr_meta.num_cols * d.d_arr_meta.max_degree) +
      (col * d.d_arr_meta.max_degree);

  int local_best = -1;

  for (int z = d.d_arr_meta.max_degree - 1 - tid; z >= 0; z -= blockDim.x) {
    if (col_data[z] > 0) {
      local_best = z;
      break;
    }
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    int other = __shfl_down_sync(0xffffffff, local_best, offset);
    if (other > local_best) local_best = other;
  }

  if (lane == 0) s_warp_best[warp_id] = local_best;
  __syncthreads();

  if (tid == 0) {
    int best_degree = -1;
    for (int w = 0; w < num_warps; w++) {
      if (s_warp_best[w] > best_degree) best_degree = s_warp_best[w];
    }

    d.d_col_best_degree[col] = best_degree;
  }
}

void update_metadata_post_ero(WPFData d, EROConfig* cERO, cudaStream_t stream) {
  const int THREADS = 256;
  const int NUM_WARPS = (THREADS + 31) / 32;

  dim3 meta_blocks = d.d_arr_meta.num_cols;
  dim3 meta_threads = 256;

  auto smem = NUM_WARPS * sizeof(int);

  _update_metadata_post_ero<<<meta_blocks, meta_threads, smem, stream>>>(d,
                                                                         cERO);
}

__global__ void _push_best_pivot(WPFData d, EROConfig* cERO) {
  auto ero = cERO[0];

  if (ero.shift < 0) return;

  int target_row = ero.target_row_idx;
  int best_degree = -1;
  int best_col = 0;

  for (int col = 0; col < d.d_arr_meta.num_cols; col++) {
    auto curr_degree = d.d_col_best_degree[col];
    if (curr_degree >= best_degree) {
      best_degree = curr_degree;
      best_col = col;
    }
  }

  if (best_degree >= 0) {
    push_stack(d.d_rows_with_pivot + (best_col * d.d_arr_meta.num_rows),
               d.d_rwp_tops[best_col], {target_row, best_degree});

    if (d.d_rwp_tops[best_col] > 1) {
      push_stack(d.d_conflicts_stack, *d.d_conflicts_stack_top, best_col);
    }
  }
}

void push_best_pivot(WPFData d, EROConfig* cERO, cudaStream_t stream) {
  _push_best_pivot<<<1, 1, 0, stream>>>(d, cERO);
}

__global__ void _check_continue(EROConfig* cERO, int* d_step,
                                int expected_steps,
                                cudaGraphConditionalHandle handle) {
  (*d_step)++;
  bool cont = (cERO[0].shift >= 0) && (*d_step < expected_steps);
  cudaGraphSetConditional(handle, cont ? 1 : 0);
}
