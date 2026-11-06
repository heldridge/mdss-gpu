#include <boost/test/unit_test.hpp>
#include <boost/test/unit_test_suite.hpp>

#include "finite_field/field.cuh"
#include "math/ff_math.h"

using Math17 = FFMath<FiniteField<17>>;
using Math100019 = FFMath<FiniteField<100019>>;
using Math1009 = FFMath<FiniteField<1009>>;

BOOST_AUTO_TEST_CASE(barycentric_weights) {
  unsigned long size = sizeof(uint32_t);
  const int num_points = 3;

  const uint32_t h_x_coords[3] = {1, 4, 11};
  uint32_t h_weights[3];

  uint32_t *d_x_coords, *d_weights;
  cudaMalloc(&d_x_coords, size * num_points);
  cudaMalloc(&d_weights, size * num_points);
  cudaMemcpy(d_x_coords, h_x_coords, num_points * size, cudaMemcpyHostToDevice);
  Math17::compute_barycentric_weights(d_x_coords, d_weights, num_points);
  cudaMemcpy(h_weights, d_weights, size * num_points, cudaMemcpyDeviceToHost);

  uint32_t correct_weights[3] = {4, 4, 9};

  BOOST_TEST(h_weights == correct_weights);
}

void run_cmp(const uint32_t *h_x_coords, uint32_t *h_M, int num_points) {
  unsigned long size = sizeof(uint32_t);
  uint32_t *d_M, *d_M_next, *d_x_coords;
  cudaMalloc(&d_M, size * (num_points + 1));
  cudaMalloc(&d_M_next, size * (num_points + 1));
  cudaMalloc(&d_x_coords, size * num_points);

  cudaMemcpy(d_x_coords, h_x_coords, num_points * size, cudaMemcpyHostToDevice);
  Math17::compute_master_polynomial(d_x_coords, d_M, num_points);
  cudaMemcpy(h_M, d_M, size * (num_points + 1), cudaMemcpyDeviceToHost);
}

BOOST_AUTO_TEST_CASE(update_M_single) {
  int num_points = 1;
  const uint32_t h_x_coords[1] = {5};
  uint32_t h_M[2] = {};
  uint32_t correct_M[2] = {12, 1};  // 17 - 5 = 12

  run_cmp(h_x_coords, h_M, num_points);

  BOOST_TEST(h_M == correct_M);
}

BOOST_AUTO_TEST_CASE(update_M_multi) {
  int num_points = 3;
  const uint32_t h_x_coords[3] = {5, 10, 13};
  uint32_t h_M[4] = {};
  uint32_t correct_M[4] = {13, 7, 6, 1};

  run_cmp(h_x_coords, h_M, num_points);

  BOOST_TEST(h_M == correct_M);
}

BOOST_AUTO_TEST_CASE(master_poly_large) {
  int num_points = 256;
  const uint32_t h_x_coords[256] = {
      641,  80,  938, 472, 923, 453, 41,  677, 450,  155, 761, 153, 289, 534,
      655,  954, 434, 576, 471, 159, 859, 86,  748,  139, 341, 657, 381, 490,
      460,  187, 59,  817, 673, 899, 111, 390, 772,  57,  812, 747, 61,  840,
      838,  461, 193, 499, 914, 996, 832, 112, 668,  532, 987, 530, 62,  389,
      418,  65,  850, 545, 2,   180, 925, 867, 881,  815, 414, 123, 432, 777,
      29,   810, 114, 269, 910, 196, 128, 995, 811,  287, 706, 600, 604, 360,
      820,  279, 417, 895, 714, 146, 119, 537, 332,  618, 302, 645, 564, 930,
      797,  945, 55,  988, 281, 258, 72,  756, 1006, 578, 71,  908, 799, 835,
      194,  712, 148, 718, 188, 394, 473, 379, 94,   458, 19,  875, 6,   789,
      3,    715, 447, 755, 192, 581, 664, 138, 102,  563, 628, 322, 496, 971,
      467,  648, 964, 505, 305, 700, 984, 124, 658,  327, 685, 17,  104, 524,
      575,  836, 920, 344, 550, 333, 531, 989, 940,  181, 927, 338, 28,  521,
      444,  331, 100, 423, 278, 504, 528, 199, 723,  250, 51,  508, 922, 656,
      612,  746, 933, 454, 448, 894, 601, 39,  905,  830, 676, 244, 555, 745,
      283,  533, 377, 651, 437, 299, 9,   809, 208,  70,  665, 705, 848, 85,
      773,  272, 387, 443, 173, 426, 929, 97,  659,  370, 547, 183, 429, 590,
      479,  191, 732, 670, 286, 973, 574, 635, 953,  947, 392, 780, 235, 726,
      1002, 699, 941, 40,  966, 401, 328, 592, 284,  238, 729, 986, 403, 638,
      918,  827, 415, 368};
  uint32_t h_M[257] = {};
  uint32_t correct_M[257] = {
      476, 443, 814, 700, 613, 32,  6,   11,   136, 705, 174, 176, 667, 713,
      83,  81,  58,  834, 994, 566, 565, 28,   897, 80,  863, 629, 729, 587,
      852, 458, 364, 815, 998, 983, 779, 585,  183, 112, 281, 680, 428, 554,
      513, 931, 896, 496, 485, 215, 120, 760,  823, 710, 652, 531, 744, 594,
      330, 547, 242, 544, 206, 127, 645, 254,  195, 804, 82,  755, 987, 860,
      325, 552, 672, 328, 686, 410, 831, 804,  193, 806, 666, 763, 347, 186,
      752, 354, 152, 551, 872, 248, 329, 936,  963, 224, 246, 423, 957, 140,
      641, 124, 364, 641, 1,   317, 225, 498,  208, 942, 235, 310, 736, 134,
      920, 432, 201, 172, 740, 778, 421, 842,  138, 482, 531, 196, 797, 554,
      730, 934, 812, 876, 430, 529, 595, 533,  680, 230, 190, 494, 877, 288,
      6,   185, 91,  532, 28,  32,  649, 947,  195, 262, 934, 390, 841, 54,
      373, 541, 114, 932, 636, 297, 311, 395,  127, 648, 833, 638, 804, 203,
      270, 436, 857, 54,  923, 838, 800, 890,  265, 420, 430, 135, 279, 522,
      535, 847, 539, 718, 64,  155, 451, 691,  916, 685, 742, 871, 218, 235,
      437, 891, 151, 717, 963, 841, 58,  37,   920, 200, 655, 422, 397, 499,
      806, 669, 769, 673, 955, 388, 915, 665,  577, 767, 524, 207, 703, 288,
      524, 390, 443, 78,  617, 306, 730, 893,  102, 710, 938, 641, 454, 0,
      213, 870, 159, 710, 873, 692, 31,  1003, 734, 585, 640, 490, 131, 380,
      345, 217, 481, 731, 1};

  unsigned long size = sizeof(uint32_t);
  uint32_t *d_M, *d_M_next, *d_x_coords;
  cudaMalloc(&d_M, size * (num_points + 1));
  cudaMalloc(&d_M_next, size * (num_points + 1));
  cudaMalloc(&d_x_coords, size * num_points);

  cudaMemcpy(d_x_coords, h_x_coords, num_points * size, cudaMemcpyHostToDevice);
  Math1009::compute_master_polynomial(d_x_coords, d_M, num_points);
  cudaMemcpy(h_M, d_M, size * (num_points + 1), cudaMemcpyDeviceToHost);
  BOOST_TEST(h_M == correct_M);
}

BOOST_AUTO_TEST_CASE(compute_b_chunk_single) {
  unsigned long size = sizeof(uint32_t);
  int num_points = 4;
  const uint32_t h_x_coords[4] = {15, 8, 14, 2};
  uint32_t h_M[5] = {};
  int S = 2;
  uint32_t h_B_chunk[8] = {};
  uint32_t correct_B_chunk[8] = {1, 1, 1, 1, 10, 3, 9, 14};
  uint32_t correct_M[5] = {11, 3, 6, 12, 1};

  uint32_t *d_b_curr, *d_M, *d_x_coords, *d_B_chunk;

  cudaMalloc(&d_M, size * (num_points + 1));
  cudaMalloc(&d_x_coords, size * num_points);
  cudaMemcpy(d_x_coords, h_x_coords, num_points * size, cudaMemcpyHostToDevice);
  cudaMalloc(&d_b_curr, size * num_points);
  cudaMalloc(&d_B_chunk, size * S * num_points);

  cudaMemcpy(d_x_coords, h_x_coords, num_points * size, cudaMemcpyHostToDevice);
  Math17::compute_master_polynomial(d_x_coords, d_M, num_points);
  cudaMemcpy(h_M, d_M, size * (num_points + 1), cudaMemcpyDeviceToHost);
  BOOST_TEST(h_M == correct_M);

  Math17::init_b_curr(d_b_curr, num_points);
  Math17::compute_b_chunk(d_b_curr, d_M, d_x_coords, d_B_chunk, num_points - 1,
                          S, num_points);
  cudaMemcpy(h_B_chunk, d_B_chunk, size * S * num_points,
             cudaMemcpyDeviceToHost);
  BOOST_TEST(h_B_chunk == correct_B_chunk);
}

BOOST_AUTO_TEST_CASE(compute_b_chunk_multi) {
  unsigned long size = sizeof(uint32_t);
  int num_points = 4;
  const uint32_t h_x_coords[4] = {15, 8, 14, 2};
  uint32_t h_M[5] = {};
  int S = 2;
  uint32_t h_B_chunk[8] = {};
  uint32_t correct_B_chunk[8] = {3, 13, 13, 0, 14, 5, 15, 3};
  uint32_t correct_M[5] = {11, 3, 6, 12, 1};

  uint32_t *d_b_curr, *d_M, *d_x_coords, *d_B_chunk;

  cudaMalloc(&d_M, size * (num_points + 1));
  cudaMalloc(&d_x_coords, size * num_points);
  cudaMemcpy(d_x_coords, h_x_coords, num_points * size, cudaMemcpyHostToDevice);
  cudaMalloc(&d_b_curr, size * num_points);
  cudaMalloc(&d_B_chunk, size * S * num_points);

  cudaMemcpy(d_x_coords, h_x_coords, num_points * size, cudaMemcpyHostToDevice);
  Math17::compute_master_polynomial(d_x_coords, d_M, num_points);
  cudaMemcpy(h_M, d_M, size * (num_points + 1), cudaMemcpyDeviceToHost);
  BOOST_TEST(h_M == correct_M);

  Math17::init_b_curr(d_b_curr, num_points);
  Math17::compute_b_chunk(d_b_curr, d_M, d_x_coords, d_B_chunk, num_points - 1,
                          S, num_points);
  Math17::compute_b_chunk(d_b_curr, d_M, d_x_coords, d_B_chunk,
                          num_points - 1 - S, S, num_points);
  cudaMemcpy(h_B_chunk, d_B_chunk, size * S * num_points,
             cudaMemcpyDeviceToHost);
  BOOST_TEST(h_B_chunk == correct_B_chunk);
}

BOOST_AUTO_TEST_CASE(compute_C_chunk) {
  unsigned long size = sizeof(uint32_t);
  int num_points = 4;
  int S = 2;
  int c = 2;
  const uint32_t h_x_coords[4] = {15, 8, 14, 2};
  const uint32_t h_y_coords[8] = {3, 1, 9, 0, 14, 6, 4, 16};
  const uint32_t correct_C[8] = {4, 2, 2, 7, 14, 15, 5, 11};
  uint32_t h_C_out[8] = {};

  uint32_t *d_b_curr, *d_M, *d_weights, *d_x_coords, *d_y_coords, *d_B_chunk,
      *d_C_out;

  cudaMalloc(&d_M, size * (num_points + 1));
  cudaMalloc(&d_x_coords, size * num_points);
  cudaMalloc(&d_y_coords, size * num_points * c);
  cudaMalloc(&d_b_curr, size * num_points);
  cudaMalloc(&d_B_chunk, size * S * num_points);
  cudaMalloc(&d_C_out, size * num_points * c);
  cudaMalloc(&d_weights, size * num_points);

  cudaMemcpy(d_x_coords, h_x_coords, num_points * size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_y_coords, h_y_coords, size * num_points * c,
             cudaMemcpyHostToDevice);
  Math17::compute_barycentric_weights(d_x_coords, d_weights, num_points);
  Math17::compute_master_polynomial(d_x_coords, d_M, num_points);

  Math17::init_b_curr(d_b_curr, num_points);
  Math17::compute_b_chunk(d_b_curr, d_M, d_x_coords, d_B_chunk, num_points - 1,
                          S, num_points);
  Math17::compute_c_chunk(d_y_coords, d_weights, d_B_chunk, d_C_out,
                          num_points - 1, S, num_points, c);
  Math17::compute_b_chunk(d_b_curr, d_M, d_x_coords, d_B_chunk,
                          num_points - 1 - S, S, num_points);
  Math17::compute_c_chunk(d_y_coords, d_weights, d_B_chunk, d_C_out,
                          num_points - 1 - S, S, num_points, c);

  cudaMemcpy(h_C_out, d_C_out, size * num_points * c, cudaMemcpyDeviceToHost);
  BOOST_TEST(h_C_out == correct_C);
}

BOOST_AUTO_TEST_CASE(langrage_coefficients_tiny) {
  unsigned long size = sizeof(uint32_t);
  int num_points = 3;
  int S = 2;
  int c = 2;
  const uint32_t h_x_coords[4] = {16, 3, 2};
  const uint32_t h_y_coords[8] = {3, 1, 7, 2, 12, 7};
  uint32_t h_lagrange_coeffs[8] = {};
  const uint32_t correct_LCs[8] = {10, 5, 15, 15, 8, 11};

  uint32_t *d_x_coords, *d_y_coords, *d_M, *d_lagrange_coeffs;

  cudaMalloc(&d_x_coords, size * num_points);
  cudaMalloc(&d_y_coords, size * num_points * c);
  cudaMalloc(&d_M, size * (num_points + 1));
  cudaMalloc(&d_lagrange_coeffs, size * num_points * c);

  cudaMemcpy(d_x_coords, h_x_coords, num_points * size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_y_coords, h_y_coords, size * num_points * c,
             cudaMemcpyHostToDevice);

  Math17::compute_master_polynomial(d_x_coords, d_M, num_points);

  Math17::compute_lagrange_coefficients(d_x_coords, d_y_coords, d_M,
                                        d_lagrange_coeffs, num_points, c, S);
  cudaMemcpy(h_lagrange_coeffs, d_lagrange_coeffs, size * num_points * c,
             cudaMemcpyDeviceToHost);
  BOOST_TEST(h_lagrange_coeffs == correct_LCs);
}

BOOST_AUTO_TEST_CASE(langrage_coefficients) {
  unsigned long size = sizeof(uint32_t);
  int num_points = 4;
  int S = 2;
  int c = 2;
  const uint32_t h_x_coords[4] = {15, 8, 14, 2};
  const uint32_t h_y_coords[8] = {3, 1, 9, 0, 14, 6, 4, 16};
  uint32_t h_lagrange_coeffs[8] = {};
  const uint32_t correct_LCs[8] = {4, 2, 2, 7, 14, 15, 5, 11};

  uint32_t *d_x_coords, *d_y_coords, *d_M, *d_lagrange_coeffs;

  cudaMalloc(&d_x_coords, size * num_points);
  cudaMalloc(&d_y_coords, size * num_points * c);
  cudaMalloc(&d_M, size * (num_points + 1));
  cudaMalloc(&d_lagrange_coeffs, size * num_points * c);

  cudaMemcpy(d_x_coords, h_x_coords, num_points * size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_y_coords, h_y_coords, size * num_points * c,
             cudaMemcpyHostToDevice);

  Math17::compute_master_polynomial(d_x_coords, d_M, num_points);

  Math17::compute_lagrange_coefficients(d_x_coords, d_y_coords, d_M,
                                        d_lagrange_coeffs, num_points, c, S);
  cudaMemcpy(h_lagrange_coeffs, d_lagrange_coeffs, size * num_points * c,
             cudaMemcpyDeviceToHost);
  BOOST_TEST(h_lagrange_coeffs == correct_LCs);
}

BOOST_AUTO_TEST_CASE(langrage_coefficients_big) {
  unsigned long size = sizeof(uint32_t);
  int num_points = 10;
  int S = 4;
  int c = 3;
  const uint32_t h_x_coords[10] = {58904, 84278, 38141, 74039, 47701,
                                   34720, 8825,  64374, 96456, 34350};
  const uint32_t h_y_coords[30] = {
      8627,  93856, 1842,  26878, 47941, 16855, 72,    86611, 67481, 15518,
      52496, 46178, 96310, 60240, 63277, 64331, 372,   46219, 19846, 78338,
      68127, 54676, 85703, 81140, 11303, 48019, 32258, 69613, 63277, 6129};
  uint32_t h_lagrange_coeffs[30] = {};
  const uint32_t correct_LCs[30] = {
      94179, 85232, 93068, 24382, 69909, 64958, 79240, 9054,  3437,  81590,
      12964, 77317, 93268, 98401, 90654, 7226,  8552,  28230, 49965, 78516,
      46145, 71199, 89967, 77960, 14173, 94711, 23613, 12801, 7341,  90341};

  uint32_t *d_x_coords, *d_y_coords, *d_M, *d_lagrange_coeffs;

  cudaMalloc(&d_x_coords, size * num_points);
  cudaMalloc(&d_y_coords, size * num_points * c);
  cudaMalloc(&d_M, size * (num_points + 1));
  cudaMalloc(&d_lagrange_coeffs, size * num_points * c);

  cudaMemcpy(d_x_coords, h_x_coords, num_points * size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_y_coords, h_y_coords, size * num_points * c,
             cudaMemcpyHostToDevice);
  Math100019::compute_master_polynomial(d_x_coords, d_M, num_points);

  Math100019::compute_lagrange_coefficients(
      d_x_coords, d_y_coords, d_M, d_lagrange_coeffs, num_points, c, S);
  cudaMemcpy(h_lagrange_coeffs, d_lagrange_coeffs, size * num_points * c,
             cudaMemcpyDeviceToHost);
  BOOST_TEST(h_lagrange_coeffs == correct_LCs);
}
