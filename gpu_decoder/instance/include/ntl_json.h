#include <NTL/ZZ.h>
#include <NTL/lzz_p.h>

#include <nlohmann/json.hpp>

namespace NTL {
inline void from_json(const nlohmann::json& j, zz_p& p) {
  auto target = j.get<long>();
  p = zz_p(target);
}

}  // namespace NTL
