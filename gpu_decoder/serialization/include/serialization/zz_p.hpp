#pragma once

#include <NTL/lzz_p.h>

#include <nlohmann/json.hpp>

namespace NTL {
inline void to_json(nlohmann::json& j, const NTL::zz_p& x) { j = NTL::rep(x); }
}  // namespace NTL
