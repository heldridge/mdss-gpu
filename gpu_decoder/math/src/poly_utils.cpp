#include "math/poly_utils.h"

#include <NTL/lzz_pX.h>

#include <cmath>
#include <stdexcept>
#include <vector>

NTL::zz_pX build_poly(std::vector<int> coeffs) {
  NTL::zz_pX poly = NTL::zz_pX();

  int i = 0;
  auto ntl_roots = NTL::Vec<NTL::zz_p>();
  for (auto r : coeffs) {
    NTL::SetCoeff(poly, i, r);
    i++;
  }
  return poly;
}

bool a_divides_b(const NTL::zz_pX &a, const NTL::zz_pX &b) {
  if (NTL::IsZero(a)) {
    throw std::invalid_argument("Divisor polynomial cannot be zero");
  }

  NTL::zz_pX quotient, remainder;
  NTL::DivRem(quotient, remainder, b, a);
  return NTL::IsZero(remainder);
}
