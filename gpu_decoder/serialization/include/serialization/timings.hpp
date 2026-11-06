#pragma once

#include "math/execution.h"
#include "mdss/execution.h"
#include "nlohmann/detail/macro_scope.hpp"
#include "nlohmann/json_fwd.hpp"
#include "serialization/zz_p.hpp"

inline void to_json(nlohmann::json& j, const WPFTiming& wt) {
  j = {{"initial_allocation", wt.initial_allocation.count()},
       {"graph_setup", wt.graph_setup.count()},
       {"main_loop", wt.main_loop.count()}};
}

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(ReconstructIterationExecution, step_config,
                                   used_batch_size, timing, found_dealer,
                                   chunk_size, expected_steps, actual_steps);

inline void to_json(nlohmann::json& j, const StepTiming& st) {
  j = {{"initial_memory", st.initial_memory.count()},
       {"master_poly", st.master_poly.count()},
       {"lagrange_coeffs", st.lagrange_coeffs.count()},
       {"build_array", st.build_array.count()},
       {"weak_popov_form_total", st.weak_popov_form_total.count()},
       {"weak_popov_form", st.weak_popov_form},
       {"convert_array", st.convert_array.count()},
       {"extract_dealer", st.extract_dealer.count()}};
}

inline void to_json(nlohmann::json& j, const ReconstructIterationTiming rit) {
  j = {{"get_subsample", rit.get_subsample.count()},
       {"step_total", rit.step_total.count()},
       {"step", rit.step},
       {"recover_value", rit.recover_value.count()},
       {"iteration_total", rit.iteration_total.count()}};
}

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(ReconstructResult, iteration_executions,
                                   values);

struct Execution {
  ReconstructResult reconstruct_result;
  bool succeeded;
  std::vector<std::vector<NTL::zz_p>> extra;
  std::vector<int> missing;
  double file_load_time;
  double reconstruct_time;
  double total_decode_time;

  NLOHMANN_DEFINE_TYPE_INTRUSIVE(Execution, reconstruct_result, succeeded,
                                 extra, missing, file_load_time,
                                 reconstruct_time, total_decode_time);
};
