#pragma once

#include <NTL/lzz_pX.h>

#include <algorithm>
#include <optional>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "config/config.h"
#include "instance/instance.h"
#include "math/ff_math.h"
#include "math/poly_matrix.h"
#include "math/poly_utils.h"
#include "mdss/execution.h"

using Clock = std::chrono::high_resolution_clock;

inline void logif(bool flag, string s) {
  if (flag) {
    std::cerr << s << std::endl;
  }
}

int calculate_t_rec(int N, int c, int degree);

uint32_t* build_array(uint32_t* d_lagrange_coeffs, uint32_t* d_M,
                      ArrayMetadata arr_meta, int poly_degree);

template <typename T>
int checked_size(const std::vector<T>& vec) {
  if (vec.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error("Vector too large");
  }
  return static_cast<int>(vec.size());
}

uint32_t convert_zz_p(NTL::zz_p val);

std::optional<std::vector<NTL::zz_pX>> extract_dealer(int degree, int lambda,
                                                      Mat<zz_pX> matrix);

template <typename Field>
StepResult mdss_step(FFMath<Field> math, int degree, int c,
                     std::vector<Point> codeword, int chunk_size,
                     bool verbose = false) {
  auto t0 = Clock::now();

  int N = checked_size(codeword);
  int t_rec = calculate_t_rec(N, c, degree);

  int lambda = degree + N - t_rec;
  auto size = sizeof(uint32_t);

  std::vector<uint32_t> h_x_coords;
  std::vector<uint32_t> h_y_coords;

  for (auto point : codeword) {
    h_x_coords.push_back(convert_zz_p(point.x));
  }

  for (auto point : codeword) {
    for (int k = 0; k < c; ++k) {
      h_y_coords.push_back(convert_zz_p(point.ys[k]));
    }
  }

  uint32_t *d_x_coords, *d_y_coords, *d_M, *d_lagrange_coeffs;
  Array3D<uint32_t> h_arr = Array3D<uint32_t>(c + 1, c + 1, N + 1);

  cudaMalloc(&d_x_coords, size * N);
  cudaMalloc(&d_y_coords, size * N * c);
  cudaMalloc(&d_M, size * (N + 1));
  cudaMalloc(&d_lagrange_coeffs, size * N * c);

  cudaMemcpy(d_x_coords, h_x_coords.data(), N * size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_y_coords, h_y_coords.data(), size * N * c,
             cudaMemcpyHostToDevice);
  auto t1 = Clock::now();

  math.compute_master_polynomial(d_x_coords, d_M, N);
  cudaDeviceSynchronize();
  auto t2 = Clock::now();

  math.compute_lagrange_coefficients(d_x_coords, d_y_coords, d_M,
                                     d_lagrange_coeffs, N, c, chunk_size);
  cudaDeviceSynchronize();
  auto t3 = Clock::now();

  ArrayMetadata d_arr_meta = {c + 1, c + 1, N + 1};
  auto d_arr = build_array(d_lagrange_coeffs, d_M, d_arr_meta, degree);
  cudaDeviceSynchronize();
  auto t4 = Clock::now();

  int expected_steps = c * (N - degree);
  auto wpf = math.weak_popov_form(d_arr, d_arr_meta, expected_steps);
  cudaDeviceSynchronize();
  auto t5 = Clock::now();

  cudaFree(d_x_coords);
  cudaFree(d_y_coords);
  cudaFree(d_M);
  cudaFree(d_lagrange_coeffs);

  cudaMemcpy(h_arr.data(), d_arr,
             d_arr_meta.num_rows * d_arr_meta.num_cols * d_arr_meta.max_degree *
                 sizeof(uint32_t),
             cudaMemcpyDeviceToHost);
  cudaFree(d_arr);

  auto matrix = array3d_to_mat(h_arr);
  auto t6 = Clock::now();

  auto result = extract_dealer(degree, lambda, matrix);
  auto t7 = Clock::now();

  logif(verbose, "Array size: " +
                     std::to_string(d_arr_meta.num_rows * d_arr_meta.num_cols *
                                    d_arr_meta.max_degree * sizeof(uint32_t)) +
                     " bytes");

  StepTiming timing = {.initial_memory = t1 - t0,
                       .master_poly = t2 - t1,
                       .lagrange_coeffs = t3 - t2,
                       .build_array = t4 - t3,
                       .weak_popov_form_total = t5 - t4,
                       .weak_popov_form = wpf.timing,
                       .convert_array = t6 - t5,
                       .extract_dealer = t7 - t6};

  return {.output = result,
          .timing = timing,
          .wpf_expected_steps = expected_steps,
          .wpf_steps_taken = wpf.steps_taken};
}

template <typename T>
std::vector<T> take_first(const std::vector<T>& vec, std::size_t count) {
  if (vec.size() < count) {
    throw std::out_of_range("vector contains fewer than requested elements");
  }

  return {vec.begin(), vec.begin() + count};
}

struct SplitMatchesResult {
  std::vector<Point> matching_points;
  std::vector<Point> non_matching_points;
};

bool point_matches(Point pt, std::vector<NTL::zz_pX> polys);

SplitMatchesResult split_matches(std::vector<Point> codeword,
                                 std::vector<NTL::zz_pX> polys);

std::vector<Point> get_subsample(std::vector<Point> all_points, int num_points,
                                 int evals_per_point);

std::vector<NTL::zz_p> get_dealer_value(
    std::vector<NTL::zz_pX> partial_dealer_polys,
    std::vector<Point> matching_points);

void print_point_vec(std::vector<Point> v);

bool terminate_decoder(TerminationCondition tc, bool past_last_step,
                       bool last_step_failed);

template <typename Field>
ReconstructResult reconstruct(FFMath<Field> math, int degree, int c,
                              std::vector<Point> codeword,
                              DecoderConfig decoder_config,
                              bool verbose = false) {
  std::vector<std::vector<NTL::zz_p>> values = {};

  bool failed = false;
  int current_step = 0;

  std::vector<ReconstructIterationExecution> executions = {};

  while (!terminate_decoder(
      decoder_config.termination_condition,
      current_step >= decoder_config.step_parameters.size(), failed)) {
    int current_step_index =
        std::min(current_step,
                 static_cast<int>(decoder_config.step_parameters.size()) - 1);

    logif(verbose, "Decoding with configuration at index: " +
                       std::to_string(current_step_index));

    auto step_config = decoder_config.step_parameters[current_step_index];

    logif(verbose,
          "Using c = " + std::to_string(step_config.step_c) +
              " and batch_size = " + std::to_string(step_config.batch_size));

    auto t0 = Clock::now();
    auto batch =
        get_subsample(codeword, step_config.batch_size, step_config.step_c);
    auto t1 = Clock::now();

    if (batch.size() < step_config.batch_size) {
      logif(verbose, "Not enough points to sample full batch size. Using " +
                         std::to_string(batch.size()) + " points instead");
    }

    StepResult step_result = mdss_step(math, degree, step_config.step_c, batch,
                                       decoder_config.chunk_size, verbose);
    auto t2 = Clock::now();

    if (step_result.output.has_value()) {
      std::vector<NTL::zz_pX> dealer_polys = step_result.output.value();
      auto sm_result = split_matches(codeword, dealer_polys);
      codeword = sm_result.non_matching_points;

      if (sm_result.matching_points.size() < decoder_config.threshold) {
        logif(verbose, "Found a dealer, but it had too few points: " +
                           std::to_string(sm_result.matching_points.size()));
        failed = true;
      } else {
        auto dealer_value =
            get_dealer_value(dealer_polys, sm_result.matching_points);
        values.push_back(dealer_value);
      }

    } else {
      failed = true;
    }
    auto t3 = Clock::now();

    ReconstructIterationTiming rit = {.get_subsample = t1 - t0,
                                      .step_total = t2 - t1,
                                      .step = step_result.timing,
                                      .recover_value = t3 - t2,
                                      .iteration_total = t3 - t0};

    ReconstructIterationExecution rie = {
        .step_config = step_config,
        .used_batch_size = checked_size(batch),
        .timing = rit,
        .found_dealer = step_result.output.has_value(),
        .chunk_size = decoder_config.chunk_size,
        .expected_steps = step_result.wpf_expected_steps,
        .actual_steps = step_result.wpf_steps_taken};

    executions.push_back(rie);

    std::ostringstream oss;
    oss << "---- WPF Timings ----" << std::endl
        << "Initial allocation:\t"
        << rit.step.weak_popov_form.initial_allocation.count() << std::endl
        << "Graph setup:\t\t" << rit.step.weak_popov_form.graph_setup.count()
        << std::endl
        << "Main loop:\t\t" << rit.step.weak_popov_form.main_loop.count()
        << std::endl
        << std::endl
        << "---- Step Timings ---- " << std::endl
        << "Get subsample:\t\t" << rit.get_subsample.count() << std::endl
        << "Initial allocation:\t" << rit.step.initial_memory.count()
        << std::endl
        << "Master poly:\t\t" << rit.step.master_poly.count() << std::endl
        << "Lagrange basis:\t\t" << rit.step.lagrange_coeffs.count()
        << std::endl
        << "Build array:\t\t" << rit.step.build_array.count() << std::endl
        << "WPF:\t\t\t" << rit.step.weak_popov_form_total.count() << std::endl
        << "Convert array:\t\t" << rit.step.convert_array.count() << std::endl
        << "Extract dealer:\t\t" << rit.step.extract_dealer.count() << std::endl
        << "Recover value:\t\t" << rit.recover_value.count() << std::endl
        << "----------" << std::endl
        << (rie.found_dealer ? "Recovered a dealer!"
                             : "Failed to recover a dealer")
        << std::endl
        << "Decoded in " << rit.iteration_total.count() << std::endl
        << "Expected Steps:\t\t" << rie.expected_steps << std::endl
        << "Actual Steps:\t\t" << rie.actual_steps << std::endl;

    logif(verbose, oss.str());

    current_step++;
  }

  return {.iteration_executions = executions, .values = values};
}
