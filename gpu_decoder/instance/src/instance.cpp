#include "instance/instance.h"

#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>

Instance load_instance(const std::string &filepath) {
  std::ifstream file(filepath);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open config file: " + filepath);
  }

  nlohmann::json j;
  file >> j;
  return j.get<Instance>();
}
