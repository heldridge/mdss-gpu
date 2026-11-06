#include "math/poly_matrix.h"

#include <NTL/lzz_pX.h>
#include <NTL/matrix.h>
#include <NTL/vec_lzz_p.h>

#include <boost/test/tools/interface.hpp>
#include <boost/test/unit_test.hpp>
#include <boost/test/unit_test_suite.hpp>

#include "math/poly_utils.h"

NTL::Mat<NTL::zz_pX> build_matrix(
    std::vector<std::vector<std::vector<int>>> root_mat) {
  auto matrix = NTL::Mat<NTL::zz_pX>();
  matrix.SetDims(root_mat.size(), root_mat[0].size());

  int i = 0;
  for (auto& row : root_mat) {
    int j = 0;
    for (auto& col : row) {
      matrix[i][j] = build_poly(root_mat[i][j]);
      j++;
    }
    i++;
  }

  return matrix;
}

BOOST_AUTO_TEST_CASE(row_norm) {
  NTL::zz_p::init(17);
  NTL::zz_pX A = build_poly({1, 3, 5});
  auto B = build_poly({2, 4, 8, 16});
  auto C = build_poly({10, 11, 12});

  NTL::Vec<NTL::zz_pX> v;
  v.append(A);
  v.append(B);
  v.append(C);

  int result = get_row_norm(v);
  BOOST_TEST(result == 3);
}

BOOST_AUTO_TEST_CASE(sv_len) {
  NTL::zz_p::init(17);

  auto matrix = build_matrix({{{5, 3, 4}, {1, 2, 3}, {16}},
                              {{1, 0, 14, 8}, {10, 11}, {9}},
                              {{4, 1}, {2, 3}, {12, 13}}});
  int result = get_length_of_shortest_row(matrix);
  BOOST_TEST(result == 1);
}

BOOST_AUTO_TEST_CASE(sv_len_2) {
  NTL::zz_p::init(17);
  auto matrix = build_matrix({{{0, 0, 1, 0, 0, 0}, {12, 6, 5, 0, 0, 0}},
                              {{0, 0, 0, 0, 0, 10}, {0, 12, 7, 14, 10}}});

  int result = get_length_of_shortest_row(matrix);
  BOOST_TEST(result == 2);
}

BOOST_AUTO_TEST_CASE(add_rows) {
  NTL::zz_p::init(17);

  auto matrix = build_matrix({{{5, 3, 4}, {1, 2, 3}, {16}},
                              {{1, 0, 14}, {10, 11}, {9}},
                              {{4, 1}, {2, 3}, {12, 13}}});

  NTL::Vec<NTL::zz_pX> correct_vec;
  correct_vec.append(build_poly({6, 3, 1}));
  correct_vec.append(build_poly({11, 13, 3}));
  correct_vec.append(build_poly({8}));

  auto result = add_rows_with_norm(matrix, 2);

  BOOST_TEST(result == correct_vec);
}
