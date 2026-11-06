#pragma once

#include <NTL/lzz_pX.h>

#include <chrono>
#include <optional>
#include <ratio>
#include <vector>

#include "config/config.h"
#include "math/execution.h"

using Duration = std::chrono::duration<double, std::milli>;

struct StepTiming {
  Duration initial_memory;
  Duration master_poly;
  Duration lagrange_coeffs;
  Duration build_array;
  Duration weak_popov_form_total;
  WPFTiming weak_popov_form;
  Duration convert_array;
  Duration extract_dealer;
};

struct StepResult {
  std::optional<std::vector<NTL::zz_pX>> output;
  StepTiming timing;
  int wpf_expected_steps;
  int wpf_steps_taken;
};

struct ReconstructIterationTiming {
  Duration get_subsample;
  Duration step_total;
  StepTiming step;
  Duration recover_value;
  Duration iteration_total;
};

struct ReconstructIterationExecution {
  StepParameters step_config;
  int used_batch_size;
  ReconstructIterationTiming timing;
  bool found_dealer;
  int chunk_size;
  int expected_steps;
  int actual_steps;
};

struct ReconstructResult {
  std::vector<ReconstructIterationExecution> iteration_executions;
  std::vector<std::vector<NTL::zz_p>> values;
};
