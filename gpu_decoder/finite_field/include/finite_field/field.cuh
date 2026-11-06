#pragma once
#include <cstdint>

template <uint32_t MODULUS>

struct FiniteField {
  __device__ __host__ static inline uint32_t add(uint32_t a, uint32_t b) {
    uint32_t s = a + b;
    return s >= MODULUS ? s - MODULUS : s;
  }

  __device__ __host__ static inline uint32_t sub(uint32_t a, uint32_t b) {
    return a >= b ? a - b : a + MODULUS - b;
  }

  __device__ __host__ static inline uint32_t mul(uint32_t a, uint32_t b) {
    return (uint32_t)(((uint64_t)a * b) % MODULUS);
  }

  __device__ __host__ static inline uint32_t pow(uint64_t base, uint32_t exp) {
    uint64_t res = 1;
    base %= MODULUS;

    while (exp > 0) {
      if (exp % 2 == 1) res = (res * base) % MODULUS;
      base = (base * base) % MODULUS;
      exp /= 2;
    }
    return (uint32_t)res;
  }

  __device__ __host__ static inline uint32_t inverse(uint32_t n) {
    return pow(n, MODULUS - 2);
  }

  __device__ __host__ static inline uint32_t add_inverse(uint32_t n) {
    return MODULUS - n;
  }
};
