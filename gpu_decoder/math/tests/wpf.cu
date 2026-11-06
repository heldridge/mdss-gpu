#include <boost/test/unit_test.hpp>
#include <boost/test/unit_test_suite.hpp>
#include <cstdint>
#include <cstring>

#include "array3d/array3d.h"
#include "finite_field/field.cuh"
#include "math/ff_math.h"

using Math17 = FFMath<FiniteField<17>>;
using Math100019 = FFMath<FiniteField<100019>>;

Array3D<uint32_t> get_array() {
  /* arr:
   * P(x) = x^3 + 2x^2 + 14x + 15
   * Row 0: [x, x^2 + 8x + 3, x^2 + 16x + 1]
   * Row 1: [0, P(x), 0]
   * Row 2: [0, 0, P(x)]
   * ----------------------
   * [0, 1, 0, 0], [3, 8, 1, 0],   [1, 16, 1, 0]
   * [0, 0, 0, 0], [15, 14, 2, 1], [0, 0, 0, 0]
   * [0, 0, 0, 0], [0, 0, 0, 0],   [15, 14, 2, 1]
   */

  Array3D<uint32_t> arr = Array3D<uint32_t>(3, 3, 4);
  int num_rows = arr.dim_x();
  int num_cols = arr.dim_y();
  int max_degree = arr.dim_z();
  uint32_t p1[4] = {3, 8, 1, 0};
  uint32_t p2[4] = {1, 16, 1, 0};
  uint32_t P[4] = {15, 14, 2, 1};

  for (int i = 0; i < num_rows; i++) {
    for (int j = 0; j < num_cols; j++) {
      for (int k = 0; k < max_degree; k++) {
        arr(i, j, k) = 0;

        if (i == 0) {
          if (j == 0) {
            if (k == 1) {
              arr(i, j, k) = 1;
            }
          } else if (j == 1) {
            arr(i, j, k) = p1[k];
          } else if (j == 2) {
            arr(i, j, k) = p2[k];
          }
        } else if (i == j) {
          arr(i, j, k) = P[k];
        }
      }
    }
  }

  return arr;
}

Array3D<uint32_t> get_big_array() {
  uint32_t arr_data[6][6][8] = {
      {{0, 0, 0, 1, 0, 0, 0, 0},
       {93438, 7202, 10545, 96725, 57561, 1686, 40742, 49353},
       {16126, 96168, 57595, 15459, 18214, 81983, 95246, 62120},
       {93779, 63762, 3438, 76425, 54810, 36368, 83316, 65191},
       {40382, 70410, 57783, 5557, 26386, 53089, 41172, 97852},
       {38454, 38773, 90725, 29945, 17338, 95897, 13933, 52491}},
      {{0, 0, 0, 0, 0, 0, 0, 0},
       {3146, 90106, 72988, 72552, 79492, 27835, 63938, 87960},
       {0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0}},
      {{0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0},
       {3146, 90106, 72988, 72552, 79492, 27835, 63938, 87960},
       {0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0}},
      {{0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0},
       {3146, 90106, 72988, 72552, 79492, 27835, 63938, 87960},
       {0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0}},
      {{0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0},
       {3146, 90106, 72988, 72552, 79492, 27835, 63938, 87960},
       {0, 0, 0, 0, 0, 0, 0, 0}},
      {{0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0},
       {3146, 90106, 72988, 72552, 79492, 27835, 63938, 87960}}};

  auto arr = Array3D<uint32_t>(6, 6, 8);
  for (int i = 0; i < 6; i++) {
    for (int j = 0; j < 6; j++) {
      for (int k = 0; k < 8; k++) {
        arr(i, j, k) = arr_data[i][j][k];
      }
    }
  }
  return arr;
}

struct DArr {
  uint32_t *d_arr;
  ArrayMetadata d_arr_meta;
};

DArr get_d_arr() {
  auto arr = get_array();
  int num_rows = arr.dim_x();
  int num_cols = arr.dim_y();
  int max_degree = arr.dim_z();

  uint32_t *d_arr;
  auto array_size = num_rows * num_cols * max_degree * sizeof(uint32_t);

  cudaMalloc(&d_arr, array_size);
  cudaMemcpy(d_arr, arr.data(), array_size, cudaMemcpyHostToDevice);

  return {d_arr, {num_rows, num_cols, max_degree}};
}

BOOST_AUTO_TEST_CASE(metadata_init) {
  auto [d_arr, d_arr_meta] = get_d_arr();

  auto data = init_wpf_data_structures(d_arr, d_arr_meta);

  init_metadata(data);

  gpuPivotInfo h_rwp[9];
  int h_conflicts_stack[3];
  int h_conflicts_stack_top[1];
  int h_rwp_tops[3];

  cudaMemcpy(h_rwp, data.d_rows_with_pivot,
             data.d_arr_meta.num_cols * data.d_arr_meta.num_rows *
                 sizeof(gpuPivotInfo),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(h_conflicts_stack, data.d_conflicts_stack,
             data.d_arr_meta.num_cols * sizeof(int), cudaMemcpyDeviceToHost);

  cudaMemcpy(h_conflicts_stack_top, data.d_conflicts_stack_top, sizeof(int),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(h_rwp_tops, data.d_rwp_tops,
             data.d_arr_meta.num_cols * sizeof(int), cudaMemcpyDeviceToHost);

  gpuPivotInfo correct_rwp[9] = {{0, 0}, {0, 0}, {0, 0}, {1, 3}, {0, 0},
                                 {0, 0}, {0, 2}, {2, 3}, {0, 0}};

  for (int i = 0; i < 9; i++) {
    BOOST_TEST(h_rwp[i].row_idx == correct_rwp[i].row_idx);
    BOOST_TEST(h_rwp[i].degree == correct_rwp[i].degree);
  }
  int correct_rwp_tops[3] = {0, 1, 2};
  BOOST_TEST(h_rwp_tops == correct_rwp_tops);

  int correct_conflicts_stack[3] = {2, 0, 0};
  BOOST_TEST(h_conflicts_stack == correct_conflicts_stack);
  BOOST_TEST(h_conflicts_stack_top[0] == 1);
}

BOOST_AUTO_TEST_CASE(set_ero) {
  auto [d_arr, d_arr_meta] = get_d_arr();
  auto d = init_wpf_data_structures(d_arr, d_arr_meta);

  EROConfig *cERO;
  EROConfig h_cERO[1] = {};

  cudaMalloc(&cERO, sizeof(EROConfig));

  init_metadata(d);
  Math17::set_current_ero(d, cERO, 0);
  cudaMemcpy(h_cERO, cERO, sizeof(EROConfig), cudaMemcpyDeviceToHost);

  EROConfig correct_cERO = {2, 0, 16, 1};

  BOOST_TEST(h_cERO[0] == correct_cERO);
}

BOOST_AUTO_TEST_CASE(perform_ero) {
  auto [d_arr, d_arr_meta] = get_d_arr();
  auto d = init_wpf_data_structures(d_arr, d_arr_meta);

  EROConfig *cERO;

  cudaMalloc(&cERO, sizeof(EROConfig));

  init_metadata(d);
  Math17::set_current_ero(d, cERO, 0);
  Math17::perform_current_ero(d.d_arr, d.d_arr_meta, cERO, 0);

  Array3D<uint32_t> h_arr = Array3D<uint32_t>(3, 3, 4);

  cudaMemcpy(h_arr.data(), d.d_arr,
             d.d_arr_meta.num_rows * d.d_arr_meta.num_cols *
                 d.d_arr_meta.max_degree * sizeof(uint32_t),
             cudaMemcpyDeviceToHost);

  uint32_t correct_arr[3][3][4] = {
      {{0, 1, 0, 0}, {3, 8, 1, 0}, {1, 16, 1, 0}},
      {{0, 0, 0, 0}, {15, 14, 2, 1}, {0, 0, 0, 0}},
      {{0, 0, 16, 0}, {0, 14, 9, 16}, {15, 13, 3, 0}}};

  for (int i = 0; i < d.d_arr_meta.num_rows; i++) {
    for (int j = 0; j < d.d_arr_meta.num_cols; j++) {
      for (int k = 0; k < d.d_arr_meta.max_degree; k++) {
        BOOST_TEST(h_arr(i, j, k) == correct_arr[i][j][k]);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(update_metadata) {
  auto [d_arr, d_arr_meta] = get_d_arr();
  auto d = init_wpf_data_structures(d_arr, d_arr_meta);

  EROConfig *cERO;

  cudaMalloc(&cERO, sizeof(EROConfig));

  init_metadata(d);
  Math17::set_current_ero(d, cERO, 0);
  Math17::perform_current_ero(d.d_arr, d.d_arr_meta, cERO, 0);
  update_metadata_post_ero(d, cERO, 0);

  cudaDeviceSynchronize();

  int h_col_best_degree[3] = {-1, -1, -1};
  cudaError_t err =
      cudaMemcpy(h_col_best_degree, d.d_col_best_degree,
                 d.d_arr_meta.num_cols * sizeof(int), cudaMemcpyDeviceToHost);

  int correct_col_best_degree[3] = {2, 3, 2};

  BOOST_TEST(h_col_best_degree == correct_col_best_degree);
}

BOOST_AUTO_TEST_CASE(push_pivot) {
  auto [d_arr, d_arr_meta] = get_d_arr();
  auto d = init_wpf_data_structures(d_arr, d_arr_meta);

  EROConfig *cERO;
  cudaMalloc(&cERO, sizeof(EROConfig));

  init_metadata(d);
  Math17::set_current_ero(d, cERO, 0);
  Math17::perform_current_ero(d.d_arr, d.d_arr_meta, cERO, 0);
  update_metadata_post_ero(d, cERO, 0);
  push_best_pivot(d, cERO, 0);

  int h_conflicts_stack[3];
  int h_conflicts_stack_top[1];
  int h_rwp_tops[3];

  gpuPivotInfo h_rwp[9];
  cudaMemcpy(
      h_rwp, d.d_rows_with_pivot,
      d.d_arr_meta.num_cols * d.d_arr_meta.num_rows * sizeof(gpuPivotInfo),
      cudaMemcpyDeviceToHost);
  cudaMemcpy(h_conflicts_stack, d.d_conflicts_stack,
             d.d_arr_meta.num_cols * sizeof(int), cudaMemcpyDeviceToHost);

  cudaMemcpy(h_conflicts_stack_top, d.d_conflicts_stack_top, sizeof(int),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(h_rwp_tops, d.d_rwp_tops, d.d_arr_meta.num_cols * sizeof(int),
             cudaMemcpyDeviceToHost);

  gpuPivotInfo correct_rwp[9] = {{0, 0}, {0, 0}, {0, 0}, {1, 3}, {2, 3},
                                 {0, 0}, {0, 2}, {2, 3}, {0, 0}};

  for (int i = 0; i < 9; i++) {
    BOOST_TEST(h_rwp[i].row_idx == correct_rwp[i].row_idx);
    BOOST_TEST(h_rwp[i].degree == correct_rwp[i].degree);
  }

  int correct_rwp_tops[3] = {0, 2, 1};
  BOOST_TEST(h_rwp_tops == correct_rwp_tops);

  int correct_conflicts_stack[3] = {1, 0, 0};
  BOOST_TEST(h_conflicts_stack == correct_conflicts_stack);
  BOOST_TEST(h_conflicts_stack_top[0] == 1);
}

BOOST_AUTO_TEST_CASE(wpf) {
  auto [d_arr, d_arr_meta] = get_d_arr();
  auto d = init_wpf_data_structures(d_arr, d_arr_meta);

  int expected_steps = 4;

  Math17::weak_popov_form(d_arr, d_arr_meta, expected_steps);

  Array3D<uint32_t> h_arr = Array3D<uint32_t>(3, 3, 4);
  cudaMemcpy(h_arr.data(), d_arr,
             d_arr_meta.num_rows * d_arr_meta.num_cols * d_arr_meta.max_degree *
                 sizeof(uint32_t),
             cudaMemcpyDeviceToHost);

  uint32_t correct_arr[3][3][4] = {
      {{0, 1, 0, 0}, {3, 8, 1, 0}, {1, 16, 1, 0}},
      {{0, 0, 11, 15}, {15, 9, 10, 0}, {0, 7, 15, 0}},
      {{0, 14, 16, 0}, {6, 4, 8, 0}, {12, 16, 0, 0}}};

  for (int i = 0; i < d_arr_meta.num_rows; i++) {
    for (int j = 0; j < d_arr_meta.num_cols; j++) {
      for (int k = 0; k < d_arr_meta.max_degree; k++) {
        BOOST_TEST(h_arr(i, j, k) == correct_arr[i][j][k]);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(wpf_big_field) {
  uint32_t arr_data[2][2][3] = {{{0, 1, 0}, {5536, 91598, 60227}},
                                {{0, 0, 0}, {74497, 39059, 61033}}};

  uint32_t *d_arr;
  cudaMalloc(&d_arr, 2 * 2 * 3 * sizeof(uint32_t));
  cudaMemcpy(d_arr, arr_data, 2 * 2 * 3 * sizeof(uint32_t),
             cudaMemcpyHostToDevice);
  ArrayMetadata d_arr_meta = {2, 2, 3};

  Math100019::weak_popov_form(d_arr, d_arr_meta, 2);

  uint32_t correct_arr[2][2][3] = {{{0, 1, 40454}, {5536, 45065, 0}},
                                   {{0, 60118, 0}, {24513, 81559, 0}}};

  uint32_t h_arr[2][2][3] = {};
  cudaMemcpy(h_arr, d_arr, 2 * 2 * 3 * sizeof(uint32_t),
             cudaMemcpyDeviceToHost);
  BOOST_TEST(h_arr == correct_arr);
}

BOOST_AUTO_TEST_CASE(wpf_big) {
  uint32_t arr_data[6][6][8] = {
      {{0, 1, 0, 0, 0, 0, 0, 0},
       {80809, 34177, 55310, 0, 0, 0, 0, 0},
       {97781, 63297, 42992, 0, 0, 0, 0, 0},
       {78760, 27532, 58144, 0, 0, 0, 0, 0},
       {28729, 98209, 38408, 0, 0, 0, 0, 0},
       {48285, 81337, 25764, 0, 0, 0, 0, 0}},
      {{0, 0, 0, 0, 0, 0, 0, 0},
       {78385, 99250, 74928, 18562, 77448, 29797, 56634, 20697},
       {0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0}},
      {{0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0},
       {78385, 99250, 74928, 18562, 77448, 29797, 56634, 20697},
       {0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0}},
      {{0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0},
       {78385, 99250, 74928, 18562, 77448, 29797, 56634, 20697},
       {0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0}},
      {{0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0},
       {78385, 99250, 74928, 18562, 77448, 29797, 56634, 20697},
       {0, 0, 0, 0, 0, 0, 0, 0}},
      {{0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0},
       {78385, 99250, 74928, 18562, 77448, 29797, 56634, 20697}}};

  uint32_t *d_arr;
  cudaMalloc(&d_arr, 6 * 6 * 8 * sizeof(uint32_t));
  cudaMemcpy(d_arr, arr_data, 6 * 6 * 8 * sizeof(uint32_t),
             cudaMemcpyHostToDevice);
  ArrayMetadata d_arr_meta = {6, 6, 8};

  Math100019::weak_popov_form(d_arr, d_arr_meta, 30);

  uint32_t correct_arr[6][6][8] = {
      {{0, 1, 0, 0, 0, 0, 0, 0},
       {80809, 34177, 55310, 0, 0, 0, 0, 0},
       {97781, 63297, 42992, 0, 0, 0, 0, 0},
       {78760, 27532, 58144, 0, 0, 0, 0, 0},
       {28729, 98209, 38408, 0, 0, 0, 0, 0},
       {48285, 81337, 25764, 0, 0, 0, 0, 0}},
      {{0, 0, 0, 0, 0, 0, 0, 0},
       {78385, 99250, 74928, 18562, 77448, 29797, 56634, 20697},
       {0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0}},
      {{0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0},
       {78385, 99250, 74928, 18562, 77448, 29797, 56634, 20697},
       {0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0}},
      {{0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0},
       {78385, 99250, 74928, 18562, 77448, 29797, 56634, 20697},
       {0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0}},
      {{0, 0, 0, 0, 0, 0, 28324, 14340},
       {48222, 11984, 97751, 91471, 27515, 44492, 35476, 0},
       {51236, 65410, 19190, 64126, 58154, 24786, 66171, 0},
       {65553, 27230, 97818, 41083, 80504, 86407, 98642, 0},
       {78385, 14389, 59267, 53581, 17156, 59638, 57122, 0},
       {0, 25751, 1484, 46209, 10222, 30462, 62548, 0}},
      {{0, 0, 0, 0, 0, 57304, 27077, 0},
       {83685, 16137, 70386, 54135, 22242, 74181, 41534, 0},
       {85084, 46529, 76396, 23272, 74912, 51809, 57056, 0},
       {17918, 54664, 28502, 3147, 75511, 73295, 82471, 0},
       {85020, 94183, 18855, 60359, 68043, 42882, 74686, 0},
       {78385, 99250, 74928, 18562, 75472, 44422, 0, 0}}};

  uint32_t h_arr[6][6][8] = {};
  cudaMemcpy(h_arr, d_arr, 6 * 6 * 8 * sizeof(uint32_t),
             cudaMemcpyDeviceToHost);
  BOOST_TEST(h_arr == correct_arr);
}
