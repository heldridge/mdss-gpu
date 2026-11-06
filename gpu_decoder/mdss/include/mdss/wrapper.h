#pragma once
#include <NTL/lzz_pX.h>

#include <vector>

#include "config/config.h"
#include "instance/instance.h"
#include "mdss/execution.h"

ReconstructResult reconstruct_main(int degree, int c,
                                   std::vector<Point> codeword,
                                   DecoderConfig decoder_config);

bool correct_field_size(long current_field_size);
