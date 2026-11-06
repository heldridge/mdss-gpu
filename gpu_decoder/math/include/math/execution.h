#pragma once

#include <chrono>
#include <ratio>

using Duration = std::chrono::duration<double, std::milli>;

struct WPFTiming {
  Duration initial_allocation;
  Duration graph_setup;
  Duration main_loop;
};

struct WPFResult {
  WPFTiming timing;
  int steps_taken;
};
