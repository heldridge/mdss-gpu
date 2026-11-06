#include <NTL/lzz_p.h>
#include <NTL/lzz_pX.h>

#include <chrono>
#include <iostream>
#include <ratio>
#include <string>

#include "CLI/CLI.hpp"
#include "config/config.h"
#include "instance/instance.h"
#include "mdss/execution.h"
#include "mdss/wrapper.h"
#include "serialization/timings.hpp"

using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double, std::milli>;

struct CheckResult {
  std::vector<std::vector<NTL::zz_p>> extra;
  std::vector<int> missing;
};

template <typename T>
bool contains(const std::vector<T>& v, const T& x) {
  return std::find(v.begin(), v.end(), x) != v.end();
}

CheckResult check_reconstruct_output(
    const std::vector<std::vector<NTL::zz_p>>& recon_output,
    const std::vector<Dealer>& present_dealers) {
  std::vector<std::vector<NTL::zz_p>> extra = {};
  std::vector<int> missing;

  std::vector<int> seen_dealers = {};
  for (auto& value : recon_output) {
    bool matched_dealer = false;

    for (auto& dealer : present_dealers) {
      bool matches_dealer = true;

      for (int i = 0; i < value.size(); i++) {
        if (value[i] != dealer.value[i]) {
          matches_dealer = false;
          break;
        }
      }

      if (matches_dealer) {
        seen_dealers.push_back(dealer.dealer_id);
        matched_dealer = true;
        break;
      }
    }

    if (!matched_dealer) {
      extra.push_back(value);
    }
  }

  for (auto& dealer : present_dealers) {
    if (!contains(seen_dealers, dealer.dealer_id)) {
      missing.push_back(dealer.dealer_id);
    }
  }

  return {.extra = extra, .missing = missing};
}

int main(int argc, char* argv[]) {
  auto t0 = Clock::now();

  CLI::App app{"Decodes MDSS instances"};

  std::string instance_filename;
  std::string config_filename;

  app.add_option("instance", instance_filename, "Path to the instance")
      ->required();
  app.add_option("config", config_filename, "Path to the config")->required();
  CLI11_PARSE(app, argc, argv);

  DecoderConfig decoder_config = load_config(config_filename);

  Instance inst = load_instance(instance_filename);
  auto t1 = Clock::now();
  Duration read_files = t1 - t0;
  std::cerr << "Time to read files: " << read_files.count() << "ms"
            << std::endl;

  if (!correct_field_size(inst.parameters.field_size)) {
    std::cerr << "Compiled with incorrect field size. Compiled with "
              << FIELD_MODULUS << ". Instance size is "
              << inst.parameters.field_size << "." << std::endl;
    return 1;
  }

  auto out = reconstruct_main(inst.parameters.degree, inst.parameters.c,
                              inst.codeword, decoder_config);
  auto t2 = Clock::now();
  Duration reconstruct = t2 - t1;
  Duration total_decode = t2 - t0;

  auto dvres = check_reconstruct_output(out.values,
                                        inst.debug_attributes.present_dealers);

  bool succeeded = dvres.missing.size() == 0 && dvres.extra.size() == 0;

  if (succeeded) {
    std::cerr << "Success: correctly found all present values!" << std::endl;
  } else {
    std::cerr << "Failure: incorrectly reconstructed values." << std::endl;
  }

  if (dvres.missing.size() > 0) {
    std::cerr << "Failed to recover the following dealers:" << std::endl;
    for (auto& md : dvres.missing) {
      std::cerr << md << std::endl;
    }
  }

  if (dvres.extra.size() > 0) {
    std::cerr << "Incorrectly recovered the following values:" << std::endl;
    for (auto& ev : dvres.extra) {
      std::cerr << "[";
      for (auto& p : ev) {
        std::cerr << p << ", ";
      }
      std::cerr << "]" << std::endl;
    }
  }

  Execution ex = {.reconstruct_result = out,
                  .succeeded = succeeded,
                  .extra = dvres.extra,
                  .missing = dvres.missing,
                  .file_load_time = read_files.count(),
                  .reconstruct_time = reconstruct.count(),
                  .total_decode_time = total_decode.count()};
  nlohmann::json j = ex;

  std::cout << j.dump(2);

  return 0;
}
