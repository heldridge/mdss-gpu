#include <boost/test/tools/old/interface.hpp>
#define BOOST_TEST_MODULE field

#include <boost/test/unit_test.hpp>

#include "finite_field/field.cuh"

using Field17 = FiniteField<17>;

BOOST_AUTO_TEST_CASE(addition_mod_17) {
  BOOST_CHECK_EQUAL(Field17::add(10, 5), 15);
  BOOST_CHECK_EQUAL(Field17::add(10, 10), 3);
  BOOST_CHECK_EQUAL(Field17::add(8, 9), 0);
}

BOOST_AUTO_TEST_CASE(subtraction_mod_17) {
  BOOST_CHECK_EQUAL(Field17::sub(10, 5), 5);
  BOOST_CHECK_EQUAL(Field17::sub(3, 8), 12);
  BOOST_CHECK_EQUAL(Field17::sub(10, 10), 0);
}

BOOST_AUTO_TEST_CASE(multiplication_mod_17) {
  BOOST_CHECK_EQUAL(Field17::mul(2, 4), 8);
  BOOST_CHECK_EQUAL(Field17::mul(10, 5), 16);
  BOOST_CHECK_EQUAL(Field17::mul(0, 5), 0);
}

BOOST_AUTO_TEST_CASE(pow_mod_17) {
  BOOST_CHECK_EQUAL(Field17::pow(2, 4), 16);
  BOOST_CHECK_EQUAL(Field17::pow(3, 3), 10);
  BOOST_CHECK_EQUAL(Field17::pow(11, 0), 1);
  BOOST_CHECK_EQUAL(Field17::pow(0, 20), 0);
}

BOOST_AUTO_TEST_CASE(inverse) { BOOST_CHECK_EQUAL(Field17::inverse(2), 9); }

BOOST_AUTO_TEST_CASE(asdf) { BOOST_CHECK_EQUAL(Field17::add_inverse(4), 13); }
