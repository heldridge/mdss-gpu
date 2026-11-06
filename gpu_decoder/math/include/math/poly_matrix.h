#include <NTL/lzz_pX.h>
#include <NTL/matrix.h>
#include <NTL/vec_lzz_p.h>

int get_row_norm(NTL::Vec<NTL::zz_pX> &row);
int get_length_of_shortest_row(NTL::Mat<NTL::zz_pX> &matrix);
NTL::Vec<NTL::zz_pX> add_rows_with_norm(NTL::Mat<NTL::zz_pX> &matrix, int norm);
