#include <gtest/gtest.h>

#include <converter.hpp>

TEST(ConverterTest, CToF) {
  auto celsius = t::Unit::Celsius(0);
  auto fahrenheit = t::Unit::Fahrenheit(32);
  EXPECT_EQ(fahrenheit, t::Unit::Fahrenheit(celsius));
}

TEST(ConverterTest, CircularConversion) {
  auto celsius = t::Unit::Celsius(0);
  auto fahrenheit = t::Unit::Fahrenheit(celsius);
  auto kelvin = t::Unit::Kelvin(fahrenheit);
  auto reaumur = t::Unit::Reaumur(kelvin);
  auto delisle = t::Unit::Delisle(reaumur);
  auto rankine = t::Unit::Rankine(delisle);
  auto romer = t::Unit::Romer(rankine);
  auto newton = t::Unit::Newton(romer);
  auto finalc = t::Unit::Celsius(newton);
  EXPECT_EQ(finalc, celsius);
}
