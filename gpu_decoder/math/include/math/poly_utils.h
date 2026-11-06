#include <NTL/lzz_pX.h>

#include <vector>

NTL::zz_pX build_poly(std::vector<int> coeffs);
bool a_divides_b(const NTL::zz_pX &a, const NTL::zz_pX &b);
