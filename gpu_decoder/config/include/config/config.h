#pragma once

#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "nlohmann/detail/macro_scope.hpp"

struct StepParameters {
  int batch_size;
  int step_c;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(StepParameters, batch_size, step_c);

enum class TerminationCondition {
  ALL_STEPS_DONE,
  FIRST_FAILURE,
  ALL_STEPS_DONE_OR_FIRST_FAILURE
};

NLOHMANN_JSON_SERIALIZE_ENUM(
    TerminationCondition,
    {{TerminationCondition::ALL_STEPS_DONE, "ALL_STEPS_DONE"},
     {TerminationCondition::FIRST_FAILURE, "FIRST_FAILURE"},
     {TerminationCondition::ALL_STEPS_DONE_OR_FIRST_FAILURE,
      "ALL_STEPS_DONE_OR_FIRST_FAILURE"}});

struct DecoderConfig {
  std::vector<StepParameters> step_parameters;
  int threshold;
  int chunk_size = 256;
  TerminationCondition termination_condition =
      TerminationCondition::FIRST_FAILURE;
};

DecoderConfig load_config(const std::string& filepath);

inline void from_json(const nlohmann::json& j, DecoderConfig& dc) {
  // Required params
  j.at("step_parameters").get_to(dc.step_parameters);
  j.at("threshold").get_to(dc.threshold);

  // Optional params with defaults
  dc.chunk_size = j.value("chunk_size", dc.chunk_size);
  dc.termination_condition =
      j.value("termination_condition", dc.termination_condition);
}
