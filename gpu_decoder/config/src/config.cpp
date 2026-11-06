#include "config/config.h"

#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>

DecoderConfig load_config(const std::string& filepath) {
  std::ifstream file(filepath);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open config file: " + filepath);
  }

  nlohmann::json j;
  file >> j;
  return j.get<DecoderConfig>();
}
