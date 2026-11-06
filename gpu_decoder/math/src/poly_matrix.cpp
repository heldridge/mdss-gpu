#include "math/poly_matrix.h"

#include <NTL/lzz_pX.h>
#include <NTL/matrix.h>
#include <NTL/vec_lzz_p.h>

#include <stdexcept>

int get_row_norm(NTL::Vec<NTL::zz_pX> &row) {
  long max_degree = 0;
  for (auto &poly : row) {
    long current_degree = deg(poly);
    if (current_degree > max_degree) {
      max_degree = current_degree;
    }
  }
  return max_degree;
}

int get_length_of_shortest_row(NTL::Mat<NTL::zz_pX> &matrix) {
  if (matrix.NumRows() < 1) {
    throw std::runtime_error("Empty matrix");
  }

  int sv_len = -1;
  for (long i = 0; i < matrix.NumRows(); i++) {
    NTL::Vec<NTL::zz_pX> &row_i = matrix[i];
    long current_len = get_row_norm(row_i);
    if (current_len < sv_len || sv_len == -1) {
      sv_len = current_len;
    }
  }

  return sv_len;
}

void add_to_vec(NTL::Vec<NTL::zz_pX> &target,
                const NTL::Vec<NTL::zz_pX> &to_add) {
  for (long i = 0; i < target.length(); i++) {
    add(target[i], target[i], to_add[i]);
  }
}

NTL::Vec<NTL::zz_pX> add_rows_with_norm(NTL::Mat<NTL::zz_pX> &matrix,
                                        int norm) {
  NTL::Vec<NTL::zz_pX> result;
  result.SetLength(matrix.NumCols());

  for (int i = 0; i < matrix.NumRows(); i++) {
    auto row = matrix[i];

    if (get_row_norm(row) == norm) {
      add_to_vec(result, row);
    }
  }

  return result;
}
