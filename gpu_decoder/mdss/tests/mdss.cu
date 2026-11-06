#define BOOST_TEST_MODULE mdss

#include <NTL/lzz_pX.h>

#include <boost/test/unit_test.hpp>
#include <boost/test/unit_test_suite.hpp>

#include "finite_field/field.cuh"
#include "instance/instance.h"
#include "math/ff_math.h"
#include "math/poly_utils.h"
#include "mdss/mdss.h"

using Math17 = FFMath<FiniteField<17>>;
using Math100019 = FFMath<FiniteField<100019>>;

BOOST_AUTO_TEST_CASE(get_t_rec_main) {
  int result = calculate_t_rec(10000, 20, 350);
  BOOST_TEST(result == 811);
}

BOOST_AUTO_TEST_CASE(build_array_main) {
  unsigned long size = sizeof(uint32_t);
  int num_points = 4;
  int S = 2;
  int c = 2;
  const uint32_t h_x_coords[4] = {15, 8, 14, 2};
  const uint32_t h_y_coords[8] = {3, 1, 9, 0, 14, 6, 4, 16};

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

  uint32_t h_arr[3][3][5] = {};
  uint32_t *d_arr =
      build_array(d_lagrange_coeffs, d_M, {c + 1, c + 1, num_points + 1}, 2);
  cudaMemcpy(h_arr, d_arr, (c + 1) * (c + 1) * (num_points + 1) * size,
             cudaMemcpyDeviceToHost);

  uint32_t correct_arr[3][3][5] = {
      {{0, 0, 1, 0, 0}, {4, 2, 2, 7, 0}, {14, 15, 5, 11, 0}},
      {{0, 0, 0, 0, 0}, {11, 3, 6, 12, 1}, {0, 0, 0, 0, 0}},
      {{0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {11, 3, 6, 12, 1}}};

  BOOST_TEST(h_arr == correct_arr);
}

BOOST_AUTO_TEST_CASE(build_array_big) {
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

  uint32_t h_arr[4][4][11] = {};
  uint32_t *d_arr =
      build_array(d_lagrange_coeffs, d_M, {c + 1, c + 1, num_points + 1}, 5);
  cudaMemcpy(h_arr, d_arr, (c + 1) * (c + 1) * (num_points + 1) * size,
             cudaMemcpyDeviceToHost);

  uint32_t correct_arr[4][4][11] = {
      {{0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0},
       {94179, 85232, 93068, 24382, 69909, 64958, 79240, 9054, 3437, 81590, 0},
       {12964, 77317, 93268, 98401, 90654, 7226, 8552, 28230, 49965, 78516, 0},
       {46145, 71199, 89967, 77960, 14173, 94711, 23613, 12801, 7341, 90341}},

      {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {7950, 5888, 54280, 67030, 72450, 98533, 42640, 43299, 23121, 58326, 1},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},

      {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {7950, 5888, 54280, 67030, 72450, 98533, 42640, 43299, 23121, 58326, 1},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},

      {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {7950, 5888, 54280, 67030, 72450, 98533, 42640, 43299, 23121, 58326, 1}},
  };

  BOOST_TEST(h_arr == correct_arr);
}

std::vector<NTL::zz_p> convert_vec(std::vector<int> v) {
  std::vector<NTL::zz_p> nv = {};
  for (auto &e : v) {
    nv.push_back(NTL::zz_p(e));
  }

  return nv;
}

std::vector<Point> get_codeword(std::vector<int> x_coords,
                                std::vector<std::vector<int>> y_coords) {
  std::vector<Point> result;

  for (int i = 0; i < x_coords.size(); i++) {
    std::vector<NTL::zz_p> new_y_coords = convert_vec(y_coords[i]);

    result.push_back({zz_p(x_coords[i]), new_y_coords});
  }

  return result;
}

struct SimplePoint {
  int x;
  std::vector<int> ys;
};

std::vector<Point> get_codeword(std::vector<SimplePoint> points) {
  std::vector<Point> result;

  for (auto &sp : points) {
    std::vector<NTL::zz_p> new_y_coords = convert_vec(sp.ys);
    result.push_back({zz_p(sp.x), new_y_coords});
  }

  return result;
}

BOOST_AUTO_TEST_CASE(decode_no_errors) {
  int degree = 2;
  int c = 1;

  NTL::zz_p::init(17);

  // 5*x^2 + 6*x + 12
  auto correct_poly = build_poly({12, 6, 5});
  std::vector<zz_pX> correct_result = {correct_poly};

  std::vector<int> x_coords = {5, 8, 0, 2, 1};
  std::vector<std::vector<int>> y_coords = {{14}, {6}, {12}, {10}, {6}};
  auto codeword = get_codeword(x_coords, y_coords);

  auto result =
      mdss_step(FFMath<FiniteField<17>>{}, degree, c, codeword, 2).output;

  BOOST_REQUIRE(result);
  BOOST_TEST(result.value() == correct_result);
}

BOOST_AUTO_TEST_CASE(decode_one_error) {
  int degree = 2;
  int c = 1;

  NTL::zz_p::init(17);

  // 5*x^2 + 6*x + 12
  auto correct_poly = build_poly({12, 6, 5});
  std::vector<zz_pX> correct_result = {correct_poly};

  std::vector<int> x_coords = {5, 8, 0, 2, 1};
  std::vector<std::vector<int>> y_coords = {{14}, {6}, {12}, {7}, {6}};
  auto codeword = get_codeword(x_coords, y_coords);

  auto result =
      mdss_step(FFMath<FiniteField<17>>{}, degree, c, codeword, 2).output;

  BOOST_REQUIRE(result);
  BOOST_TEST(result.value() == correct_result);
}

BOOST_AUTO_TEST_CASE(decode_too_many_errors) {
  int degree = 2;
  int c = 1;

  NTL::zz_p::init(17);

  std::vector<int> x_coords = {5, 8, 0, 2, 1};
  std::vector<std::vector<int>> y_coords = {{14}, {6}, {9}, {7}, {6}};
  auto codeword = get_codeword(x_coords, y_coords);

  auto result =
      mdss_step(FFMath<FiniteField<17>>{}, degree, c, codeword, 2).output;

  BOOST_TEST(!result.has_value());
}

BOOST_AUTO_TEST_CASE(test_point_matches) {
  NTL::zz_p::init(17);

  auto poly_1 = build_poly({16, 3, 9, 10});
  auto poly_2 = build_poly({9, 12, 2, 1});

  auto x_1 = zz_p(4);
  auto x_2 = zz_p(5);
  Point real_point = {
      x_1, {NTL::eval(poly_1, zz_p(x_1)), NTL::eval(poly_2, zz_p(x_1))}};
  Point bad_point = {
      x_2, {NTL::eval(poly_1, zz_p(x_2)), NTL::eval(poly_2, zz_p(x_2)) + 1}};

  BOOST_TEST(point_matches(real_point, {poly_1, poly_2}));
  BOOST_TEST(!point_matches(bad_point, {poly_1, poly_2}));
}

BOOST_AUTO_TEST_CASE(test_take_first) {
  std::vector<int> v = {4, 5, 6, 7, 8, 9};

  std::vector<int> corr = {4, 5, 6};

  BOOST_TEST(take_first(v, 3) == corr);
}

BOOST_AUTO_TEST_CASE(test_reconstruct_basic) {
  std::vector<Point> inst = get_codeword({
      {8, {13}},
      {5, {4}},
      {11, {0}},
  });

  std::vector<std::vector<NTL::zz_p>> correct_vals = {convert_vec({10})};

  auto result = reconstruct(FFMath<FiniteField<17>>{}, 2, 1, inst,
                            {.step_parameters = {{3, 1}}}, 2)
                    .values;

  BOOST_TEST(result == correct_vals);
}

BOOST_AUTO_TEST_CASE(test_reconstruct_b) {
  std::vector<Point> inst = get_codeword({
      {16, {3, 1}},
      {3, {7, 2}},
      {2, {12, 7}},
  });

  std::vector<std::vector<NTL::zz_p>> correct_vals = {convert_vec({10, 15})};

  auto result = reconstruct(FFMath<FiniteField<17>>{}, 2, 2, inst,
                            {.step_parameters = {{3, 2}}}, 2)
                    .values;

  BOOST_TEST(result == correct_vals);
}

BOOST_AUTO_TEST_CASE(test_reconstruct_c) {
  NTL::zz_p::init(17);

  std::vector<Point> inst = get_codeword({
      {3, {10, 15, 8}},
      {12, {15, 13, 4}},
      {7, {10, 6, 3}},
      {4, {4, 7, 4}},
      {13, {11, 14, 8}},
      {9, {0, 16, 3}},
      {2, {3, 7, 11}},
      {11, {6, 13, 16}},
      {16, {6, 6, 14}},
      {10, {1, 14, 10}},
  });

  std::vector<std::vector<NTL::zz_p>> correct_vals = {convert_vec({1, 11, 14})};

  auto result = reconstruct(FFMath<FiniteField<17>>{}, 2, 3, inst,
                            {.step_parameters = {{10, 2}}}, 2)
                    .values;

  BOOST_TEST(result == correct_vals);
}

BOOST_AUTO_TEST_CASE(test_reconstruct_d) {
  NTL::zz_p::init(100019);

  std::vector<Point> inst = get_codeword({
      {73273, {42765, 50457, 40033}},
      {65729, {88710, 14284, 23566}},
      {1126, {40424, 50949, 98244}},
      {7054, {62987, 18036, 33017}},
      {82930, {85317, 759, 34628}},
      {50464, {21398, 6020, 80509}},
      {80685, {99645, 90264, 22562}},
      {97236, {11087, 24395, 61956}},
      {2111, {51709, 46567, 10760}},
      {11434, {60600, 6647, 20140}},
  });

  std::vector<std::vector<NTL::zz_p>> correct_vals = {
      convert_vec({39364, 10609, 67415})};

  auto result = reconstruct(FFMath<FiniteField<100019>>{}, 3, 3, inst,
                            {.step_parameters = {{10, 3}}}, 2)
                    .values;
  BOOST_TEST(result == correct_vals);
}

BOOST_AUTO_TEST_CASE(test_reconstruct_e) {
  NTL::zz_p::init(100019);

  std::vector<Point> inst = get_codeword({
      {66885, {1792, 93993, 61719, 21605, 22109}},
      {76390, {31679, 23838, 95005, 66869, 47028}},
      {85871, {74004, 89464, 23830, 23822, 75078}},
      {64188, {26858, 99336, 88785, 9629, 83045}},
      {65420, {17161, 65907, 30791, 87285, 89096}},
      {32144, {82756, 34712, 96031, 50764, 96030}},
      {70931, {27056, 60682, 79630, 68442, 21263}},
      {35725, {95683, 36361, 93937, 79207, 33966}},
      {89591, {83267, 76249, 83657, 83104, 69901}},
      {22609, {80184, 22708, 30643, 41683, 49943}},
      {97865, {13106, 27361, 60752, 92493, 88209}},
      {71456, {37337, 29215, 78251, 17505, 78755}},
      {19476, {60013, 94622, 2752, 37756, 79214}},
      {18301, {40, 78783, 38393, 72861, 94448}},
      {87555, {12862, 5529, 93587, 43535, 84454}},
      {29151, {82287, 84275, 88535, 48734, 46478}},
      {61841, {63702, 63568, 6389, 54474, 99496}},
      {23017, {90747, 20225, 67911, 2889, 62978}},
  });

  std::vector<std::vector<NTL::zz_p>> correct_vals = {
      convert_vec({54162, 20238, 76210, 70826, 53511}),
      convert_vec({39827, 34917, 41960, 69616, 44342})};

  auto result = reconstruct(FFMath<FiniteField<100019>>{}, 3, 5, inst,
                            {.step_parameters = {{18, 5}}}, 3)
                    .values;

  BOOST_TEST(result == correct_vals);
}
