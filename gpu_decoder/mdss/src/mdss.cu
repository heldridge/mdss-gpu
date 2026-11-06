#include <NTL/lzz_pX.h>
#include <NTL/matrix.h>

#include <random>

#include "config/config.h"
#include "finite_field/field.cuh"
#include "math/ff_math.h"
#include "math/poly_matrix.h"
#include "math/poly_utils.h"
#include "mdss/mdss.h"

using FF = FiniteField<FIELD_MODULUS>;
using Math = FFMath<FF>;

int calculate_t_rec(int N, int c, int degree) {
  return ceil(static_cast<double>(N + (c * (degree + 1))) / (c + 1));
}

uint32_t convert_zz_p(NTL::zz_p val) {
  auto rep = NTL::rep(val);
  return static_cast<uint32_t>(rep);
}

uint32_t* build_array(uint32_t* d_lagrange_coeffs, uint32_t* d_M,
                      ArrayMetadata arr_meta, int poly_degree) {
  auto val_size = sizeof(uint32_t);
  int array_size = arr_meta.num_rows * arr_meta.num_cols * arr_meta.max_degree;
  uint32_t* d_arr;
  cudaMalloc(&d_arr, array_size * val_size);
  cudaMemset(d_arr, 0, array_size * val_size);

  std::vector<uint32_t> one_one;
  for (int z = 0; z < arr_meta.max_degree; z++) {
    if (z == poly_degree) {
      one_one.push_back(1);
    } else {
      one_one.push_back(0);
    }
  }
  cudaMemcpy(d_arr, one_one.data(), arr_meta.max_degree * val_size,
             cudaMemcpyHostToDevice);

  for (int j = 1; j < arr_meta.num_cols; j++) {
    cudaMemcpy(d_arr + (j * arr_meta.max_degree),
               d_lagrange_coeffs + ((j - 1) * (arr_meta.max_degree - 1)),
               (arr_meta.max_degree - 1) * val_size, cudaMemcpyDeviceToDevice);
  }

  for (int i = 1; i < arr_meta.num_rows; i++) {
    cudaMemcpy(d_arr + (i * arr_meta.num_cols * arr_meta.max_degree) +
                   (i * arr_meta.max_degree),
               d_M, arr_meta.max_degree * val_size, cudaMemcpyDeviceToDevice);
  }

  return d_arr;
}

std::vector<Point> get_subsample(std::vector<Point> all_points, int num_points,
                                 int evals_per_point) {
  std::mt19937 rng(std::random_device{}());
  std::vector<Point> sampled_points;
  sampled_points.reserve(num_points);

  std::sample(all_points.begin(), all_points.end(),
              std::back_inserter(sampled_points), num_points, rng);
  std::shuffle(sampled_points.begin(), sampled_points.end(), rng);

  for (auto& sp : sampled_points) {
    sp.ys = take_first(sp.ys, evals_per_point);
  }

  return sampled_points;
}

bool point_matches(Point pt, std::vector<NTL::zz_pX> polys) {
  for (int i = 0; i < polys.size(); i++) {
    if (pt.ys[i] != NTL::eval(polys[i], pt.x)) {
      return false;
    }
  }
  return true;
}

SplitMatchesResult split_matches(std::vector<Point> codeword,
                                 std::vector<NTL::zz_pX> polys) {
  std::vector<Point> matching_points = {};
  std::vector<Point> non_matching_points = {};

  for (auto& point : codeword) {
    if (point_matches(point, polys)) {
      matching_points.push_back(point);
    } else {
      non_matching_points.push_back(point);
    }
  }

  return {matching_points, non_matching_points};
}

std::vector<NTL::zz_pX> interpolate_remaining_polys(
    std::vector<NTL::zz_pX> starting_polys,
    std::vector<Point> matching_points) {
  if (matching_points.size() == 0) {
    return {};
  }

  std::vector<NTL::zz_pX> remaining_polys;

  int num_evals = matching_points[0].ys.size();

  NTL::Vec<NTL::zz_p> interp_x_coords;
  for (auto& pt : matching_points) {
    interp_x_coords.append(pt.x);
  }

  for (int j = starting_polys.size(); j < num_evals; j++) {
    NTL::Vec<NTL::zz_p> interp_y_coords;
    for (auto& pt : matching_points) {
      interp_y_coords.append(pt.ys[j]);
    }

    zz_pX interp_poly = NTL::interpolate(interp_x_coords, interp_y_coords);
    remaining_polys.push_back(interp_poly);
  }

  return remaining_polys;
}

std::vector<NTL::zz_p> get_dealer_value(
    std::vector<NTL::zz_pX> partial_dealer_polys,
    std::vector<Point> matching_points) {
  /* This function is needed as recovered dealers from iterations may not
  have
   * all of their polynomials recovered. So, we have this function to recover
   * the remaining polynomials and then stitch together the value.
   */

  auto remaining_polys =
      interpolate_remaining_polys(partial_dealer_polys, matching_points);

  std::vector<NTL::zz_p> result = {};

  auto pll = {partial_dealer_polys, remaining_polys};
  for (const auto& l : pll) {
    for (const auto& p : l) {
      result.push_back(NTL::eval(p, NTL::zz_p(0)));
    }
  }
  return result;
}

void print_point_vec(std::vector<Point> v) {
  std::cout << "[" << std::endl;
  for (int i = 0; i < v.size(); i++) {
    if (i != 0) {
      std::cout << "," << std::endl;
    }
    std::cout << v[i];
  }
  std::cout << std::endl << "]" << std::endl;
}

template ReconstructResult reconstruct<FF>(Math, int, int, std::vector<Point>,
                                           DecoderConfig, bool);

ReconstructResult reconstruct_main(int degree, int c,
                                   std::vector<Point> codeword,
                                   DecoderConfig decoder_config) {
  return reconstruct<FF>(Math{}, degree, c, codeword, decoder_config, true);
}

bool correct_field_size(long current_field_size) {
  return (current_field_size == FIELD_MODULUS);
}

std::optional<std::vector<NTL::zz_pX>> extract_dealer(int degree, int lambda,
                                                      Mat<zz_pX> matrix) {
  int sv_len = get_length_of_shortest_row(matrix);

  if (sv_len > lambda) {
    return std::nullopt;
  }

  auto sv_sum = add_rows_with_norm(matrix, sv_len);

  zz_pX divisor;
  NTL::SetCoeff(divisor, degree, 1);
  sv_sum[0] = sv_sum[0] / divisor;

  if (IsZero(sv_sum[0])) {
    return std::nullopt;
  }

  if (a_divides_b(sv_sum[0], sv_sum[1])) {
    std::vector<NTL::zz_pX> result = {};

    for (int i = 1; i < sv_sum.length(); i++) {
      result.push_back(sv_sum[i] / sv_sum[0]);
    }

    return result;
  } else {
    return std::nullopt;
  }
}

bool terminate_decoder(TerminationCondition tc, bool past_last_step,
                       bool last_step_failed) {
  switch (tc) {
    case TerminationCondition::FIRST_FAILURE:
      return last_step_failed;
    case TerminationCondition::ALL_STEPS_DONE:
      return past_last_step;
    case TerminationCondition::ALL_STEPS_DONE_OR_FIRST_FAILURE:
      return last_step_failed || past_last_step;
  }
  return true;
}
