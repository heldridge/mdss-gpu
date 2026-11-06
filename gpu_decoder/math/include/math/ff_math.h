#pragma once
#include <cstdint>

#include "array3d/array3d.h"
#include "math/execution.h"

struct gpuPivotInfo {
  int row_idx;
  int degree;
};

struct EROConfig {
  int target_row_idx;
  int source_row_idx;
  uint32_t zeroing_coefficient;
  int shift;

  bool operator==(const EROConfig&) const = default;
};

struct ArrayMetadata {
  int num_rows;
  int num_cols;
  int max_degree;
};

struct WPFData {
  uint32_t* d_arr;
  ArrayMetadata d_arr_meta;
  gpuPivotInfo* d_rows_with_pivot;
  int* d_rwp_tops;
  int* d_conflicts_stack;
  int* d_conflicts_stack_top;
  int* d_col_best_degree;
};

template <typename T>
__host__ __device__ inline void push_stack(T* stack, int& top, T value);

template <typename T>
__host__ __device__ inline T pop_stack(T* stack, int& top);

std::ostream& operator<<(std::ostream& os, const EROConfig& ec);

template <typename Field>
struct FFMath {
  static void compute_barycentric_weights(const uint32_t* d_x_coords,
                                          uint32_t* d_weights, int num_points);
  static void compute_master_polynomial(const uint32_t* d_x_coords,
                                        uint32_t*& master_poly, int num_points);

  static void compute_b_chunk(uint32_t* d_b_curr, const uint32_t* d_M,
                              const uint32_t* d_x_coords, uint32_t* d_B_chunk,
                              int j_start, int S, int num_points);
  static void init_b_curr(uint32_t* d_b_curr, int num_points);

  static void compute_c_chunk(const uint32_t* d_y_vals,
                              const uint32_t* d_weights,
                              const uint32_t* d_B_chunk, uint32_t* d_C_out,
                              int j_start, int S, int num_points, int c);

  static void compute_lagrange_coefficients(const uint32_t* d_x_coords,
                                            const uint32_t* d_y_coords,
                                            const uint32_t* d_M,
                                            uint32_t* d_lagrange_coeffs,
                                            int num_points, int c,
                                            int chunk_size);

  static WPFResult weak_popov_form(uint32_t* d_arr, ArrayMetadata d_arr_meta,
                                   int expected_steps);

  static void set_current_ero(WPFData d, EROConfig* cERO, cudaStream_t stream);

  static void perform_current_ero(uint32_t* d_arr, ArrayMetadata d_arr_meta,
                                  EROConfig* cERO, cudaStream_t stream);
};

void init_metadata(WPFData d);
WPFData init_wpf_data_structures(uint32_t* d_arr, ArrayMetadata d_arr_meta);
void update_metadata_post_ero(WPFData d, EROConfig* cERO, cudaStream_t stream);
void push_best_pivot(WPFData d, EROConfig* cERO, cudaStream_t stream);
__global__ void _check_continue(EROConfig* cERO, int* d_step,
                                int expected_steps,
                                cudaGraphConditionalHandle handle);

#include "lagrange.tpp"
#include "wpf.tpp"
