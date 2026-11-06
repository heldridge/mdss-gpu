#pragma once

#include <NTL/lzz_p.h>

#include <nlohmann/json.hpp>
#include <vector>

#include "nlohmann/detail/macro_scope.hpp"
#include "ntl_json.h"

struct InstanceParameters {
  long field_size;
  int c;
  int degree;
  int N;
  int t_rec;
};

struct Dealer {
  int dealer_id;
  int num_points;
  std::vector<NTL::zz_p> value;
};

struct DebugAttributes {
  std::vector<Dealer> present_dealers;
  bool is_nice;
};

struct Point {
  NTL::zz_p x;
  std::vector<NTL::zz_p> ys;
};

inline std::ostream& operator<<(std::ostream& os, const Point& point) {
  os << "{" << point.x << ", [";
  for (int i = 0; i < point.ys.size(); i++) {
    if (i != 0) {
      os << ", ";
    }
    os << point.ys[i];
  }
  os << "]";
  return os << "}";
}

struct Instance {
  InstanceParameters parameters;
  DebugAttributes debug_attributes;
  std::vector<Point> codeword;
};

inline void from_json(const nlohmann::json& j, Dealer& d) {
  j.at("dealer_id").get_to(d.dealer_id);
  j.at("num_points").get_to(d.num_points);

  d.value.clear();
  std::vector<std::vector<NTL::zz_p>> polynomials =
      j.at("polynomials").get<std::vector<std::vector<NTL::zz_p>>>();

  for (const auto& poly_coeffs : polynomials) {
    d.value.push_back(poly_coeffs[0]);
  }
}

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(InstanceParameters, field_size, c, degree, N,
                                   t_rec);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(DebugAttributes, present_dealers, is_nice)

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Point, x, ys);

inline void from_json(const nlohmann::json& j, Instance& i) {
  j.at("parameters").get_to(i.parameters);

  NTL::zz_p::init(i.parameters.field_size);
  j.at("debug_attributes").get_to(i.debug_attributes);
  j.at("codeword").get_to(i.codeword);
}

Instance load_instance(const std::string& filepath);
