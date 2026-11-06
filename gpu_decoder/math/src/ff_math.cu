#include "finite_field/field.cuh"
#include "math/ff_math.h"
// Instantiate one close to the one for 100k (100003), but not exactly so that
// we don't get duplicate definitions.
extern template struct FFMath<FiniteField<100019>>;
extern template struct FFMath<FiniteField<17>>;
extern template struct FFMath<FiniteField<FIELD_MODULUS>>;
