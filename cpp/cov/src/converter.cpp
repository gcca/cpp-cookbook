#include <ostream>

#include "converter.hpp"

namespace t {

namespace {

class UnitBase : public Unit {
public:
  constexpr explicit UnitBase(double value) : value_{value} {}

  double value() const noexcept final { return value_; }

  virtual std::unique_ptr<Unit> ToCelsius() const noexcept = 0;
  virtual std::unique_ptr<Unit> ToFahrenheit() const noexcept = 0;
  virtual std::unique_ptr<Unit> ToKelvin() const noexcept = 0;
  virtual std::unique_ptr<Unit> ToReaumur() const noexcept = 0;
  virtual std::unique_ptr<Unit> ToDelisle() const noexcept = 0;
  virtual std::unique_ptr<Unit> ToRankine() const noexcept = 0;
  virtual std::unique_ptr<Unit> ToRomer() const noexcept = 0;
  virtual std::unique_ptr<Unit> ToNewton() const noexcept = 0;

private:
  double value_;
};

class Celsius : public UnitBase {
public:
  using UnitBase::UnitBase;

  const std::string & name() const noexcept final {
    static const std::string name = "Celsius";
    return name;
  }

  std::unique_ptr<Unit> ToCelsius() const noexcept final {
    return Unit::Celsius(value());
  }

  std::unique_ptr<Unit> ToFahrenheit() const noexcept final {
    return Unit::Fahrenheit(value() * 9 / 5 + 32);
  }

  std::unique_ptr<Unit> ToKelvin() const noexcept final {
    return Unit::Kelvin(value() + 273.15);
  }

  std::unique_ptr<Unit> ToReaumur() const noexcept final {
    return Unit::Reaumur(value() * 4 / 5);
  }

  std::unique_ptr<Unit> ToDelisle() const noexcept {
    return Unit::Delisle((100 - value()) * 3 / 2);
  }

  std::unique_ptr<Unit> ToRankine() const noexcept {
    return Unit::Rankine((value() + 273.15) * 9 / 5);
  }

  std::unique_ptr<Unit> ToRomer() const noexcept {
    return Unit::Romer(value() * 21 / 40 + 7.5);
  }

  std::unique_ptr<Unit> ToNewton() const noexcept {
    return Unit::Newton(value() * 33 / 100);
  }
};

class Fahrenheit : public UnitBase {
public:
  using UnitBase::UnitBase;

  const std::string & name() const noexcept final {
    static const std::string name = "Fahrenheit";
    return name;
  }

  std::unique_ptr<Unit> ToCelsius() const noexcept final {
    return Unit::Celsius((value() - 32) * 5 / 9);
  }

  std::unique_ptr<Unit> ToFahrenheit() const noexcept final {
    return Unit::Fahrenheit(value());
  }

  std::unique_ptr<Unit> ToKelvin() const noexcept final {
    return Unit::Kelvin((value() + 459.67) * 5 / 9);
  }

  std::unique_ptr<Unit> ToReaumur() const noexcept final {
    return Unit::Reaumur((value() - 32) * 4 / 9);
  }

  std::unique_ptr<Unit> ToDelisle() const noexcept {
    return Unit::Delisle((212 - value()) * 5 / 6);
  }

  std::unique_ptr<Unit> ToRankine() const noexcept {
    return Unit::Rankine(value() + 459.67);
  }

  std::unique_ptr<Unit> ToRomer() const noexcept {
    return Unit::Romer((value() - 32) * 7 / 24 + 7.5);
  }

  std::unique_ptr<Unit> ToNewton() const noexcept {
    return Unit::Newton((value() - 32) * 11 / 60);
  }
};

class Kelvin : public UnitBase {
public:
  using UnitBase::UnitBase;

  const std::string & name() const noexcept final {
    static const std::string name = "Kelvin";
    return name;
  }

  std::unique_ptr<Unit> ToCelsius() const noexcept final {
    return Unit::Celsius(value() - 273.15);
  }

  std::unique_ptr<Unit> ToFahrenheit() const noexcept final {
    return Unit::Fahrenheit(value() * 9 / 5 - 459.67);
  }

  std::unique_ptr<Unit> ToKelvin() const noexcept final {
    return Unit::Kelvin(value());
  }

  std::unique_ptr<Unit> ToReaumur() const noexcept final {
    return Unit::Reaumur((value() - 273.15) * 4 / 5);
  }

  std::unique_ptr<Unit> ToDelisle() const noexcept {
    return Unit::Delisle((373.15 - value()) * 3 / 2);
  }

  std::unique_ptr<Unit> ToRankine() const noexcept {
    return Unit::Rankine(value() * 9 / 5);
  }

  std::unique_ptr<Unit> ToRomer() const noexcept {
    return Unit::Romer((value() - 273.15) * 21 / 40 + 7.5);
  }

  std::unique_ptr<Unit> ToNewton() const noexcept {
    return Unit::Newton((value() - 273.15) * 33 / 100);
  }
};

class Reaumur : public UnitBase {
public:
  using UnitBase::UnitBase;

  const std::string & name() const noexcept final {
    static const std::string name = "Reaumur";
    return name;
  }

  std::unique_ptr<Unit> ToCelsius() const noexcept final {
    return Unit::Celsius(value() * 5 / 4);
  }

  std::unique_ptr<Unit> ToFahrenheit() const noexcept final {
    return Unit::Fahrenheit(value() * 9 / 4 + 32);
  }

  std::unique_ptr<Unit> ToKelvin() const noexcept final {
    return Unit::Kelvin(value() * 5 / 4 + 273.15);
  }

  std::unique_ptr<Unit> ToReaumur() const noexcept final {
    return Unit::Reaumur(value());
  }

  std::unique_ptr<Unit> ToDelisle() const noexcept {
    return Unit::Delisle((80 - value()) * 15 / 8);
  }

  std::unique_ptr<Unit> ToRankine() const noexcept {
    return Unit::Rankine(value() * 9 / 4 + 491.67);
  }

  std::unique_ptr<Unit> ToRomer() const noexcept {
    return Unit::Romer(value() * 21 / 32 + 7.5);
  }

  std::unique_ptr<Unit> ToNewton() const noexcept {
    return Unit::Newton(value() * 33 / 60);
  }
};

class Delisle : public UnitBase {
public:
  using UnitBase::UnitBase;

  const std::string & name() const noexcept final {
    static const std::string name = "Delisle";
    return name;
  }

  std::unique_ptr<Unit> ToCelsius() const noexcept final {
    return Unit::Celsius(100 - value() * 2 / 3);
  }

  std::unique_ptr<Unit> ToFahrenheit() const noexcept final {
    return Unit::Fahrenheit(212 - value() * 6 / 5);
  }

  std::unique_ptr<Unit> ToKelvin() const noexcept final {
    return Unit::Kelvin(373.15 - value() * 2 / 3);
  }

  std::unique_ptr<Unit> ToReaumur() const noexcept final {
    return Unit::Reaumur(80 - value() * 8 / 15);
  }

  std::unique_ptr<Unit> ToDelisle() const noexcept {
    return Unit::Delisle(value());
  }

  std::unique_ptr<Unit> ToRankine() const noexcept {
    return Unit::Rankine(671.67 - value() * 6 / 5);
  }

  std::unique_ptr<Unit> ToRomer() const noexcept {
    return Unit::Romer(60 - value() * 7 / 20);
  }

  std::unique_ptr<Unit> ToNewton() const noexcept {
    return Unit::Newton(33 - value() * 11 / 50);
  }
};

class Rankine : public UnitBase {
public:
  using UnitBase::UnitBase;

  const std::string & name() const noexcept final {
    static const std::string name = "Rankine";
    return name;
  }

  std::unique_ptr<Unit> ToCelsius() const noexcept final {
    return Unit::Celsius((value() - 491.67) * 5 / 9);
  }

  std::unique_ptr<Unit> ToFahrenheit() const noexcept final {
    return Unit::Fahrenheit(value() - 459.67);
  }

  std::unique_ptr<Unit> ToKelvin() const noexcept final {
    return Unit::Kelvin(value() * 5 / 9);
  }

  std::unique_ptr<Unit> ToReaumur() const noexcept final {
    return Unit::Reaumur((value() - 491.67) * 4 / 9);
  }

  std::unique_ptr<Unit> ToDelisle() const noexcept {
    return Unit::Delisle((671.67 - value()) * 5 / 6);
  }

  std::unique_ptr<Unit> ToRankine() const noexcept {
    return Unit::Rankine(value());
  }

  std::unique_ptr<Unit> ToRomer() const noexcept {
    return Unit::Romer((value() - 491.67) * 7 / 24 + 7.5);
  }

  std::unique_ptr<Unit> ToNewton() const noexcept {
    return Unit::Newton((value() - 491.67) * 11 / 60);
  }
};

class Romer : public UnitBase {
public:
  using UnitBase::UnitBase;

  const std::string & name() const noexcept final {
    static const std::string name = "Romer";
    return name;
  }

  std::unique_ptr<Unit> ToCelsius() const noexcept final {
    return Unit::Celsius((value() - 7.5) * 40 / 21);
  }

  std::unique_ptr<Unit> ToFahrenheit() const noexcept final {
    return Unit::Fahrenheit((value() - 7.5) * 24 / 7 + 32);
  }

  std::unique_ptr<Unit> ToKelvin() const noexcept final {
    return Unit::Kelvin((value() - 7.5) * 40 / 21 + 273.15);
  }

  std::unique_ptr<Unit> ToReaumur() const noexcept final {
    return Unit::Reaumur((value() - 7.5) * 32 / 21);
  }

  std::unique_ptr<Unit> ToDelisle() const noexcept {
    return Unit::Delisle((60 - value()) * 20 / 7);
  }

  std::unique_ptr<Unit> ToRankine() const noexcept {
    return Unit::Rankine((value() - 7.5) * 24 / 7 + 491.67);
  }

  std::unique_ptr<Unit> ToRomer() const noexcept {
    return Unit::Romer(value());
  }

  std::unique_ptr<Unit> ToNewton() const noexcept {
    return Unit::Newton((value() - 7.5) * 22 / 35);
  }
};

class Newton : public UnitBase {
public:
  using UnitBase::UnitBase;

  const std::string & name() const noexcept final {
    static const std::string name = "Newton";
    return name;
  }

  std::unique_ptr<Unit> ToCelsius() const noexcept final {
    return Unit::Celsius(value() * 100 / 33);
  }

  std::unique_ptr<Unit> ToFahrenheit() const noexcept final {
    return Unit::Fahrenheit(value() * 60 / 11 + 32);
  }

  std::unique_ptr<Unit> ToKelvin() const noexcept final {
    return Unit::Kelvin(value() * 100 / 33 + 273.15);
  }

  std::unique_ptr<Unit> ToReaumur() const noexcept final {
    return Unit::Reaumur(value() * 80 / 33);
  }

  std::unique_ptr<Unit> ToDelisle() const noexcept {
    return Unit::Delisle((33 - value()) * 50 / 11);
  }

  std::unique_ptr<Unit> ToRankine() const noexcept {
    return Unit::Rankine(value() * 60 / 11 + 491.67);
  }

  std::unique_ptr<Unit> ToRomer() const noexcept {
    return Unit::Romer(value() * 35 / 22 + 7.5);
  }

  std::unique_ptr<Unit> ToNewton() const noexcept {
    return Unit::Newton(value());
  }
};

template <std::unique_ptr<Unit> (UnitBase::*Converter)() const noexcept>
static std::unique_ptr<Unit> Convert(const std::unique_ptr<Unit> & unit) {
  UnitBase * base = dynamic_cast<UnitBase *>(unit.get());
  if (base) { return (base->*Converter)(); }
  throw std::bad_cast{};
}

}  // namespace

std::ostream & operator<<(std::ostream & os,
                          const std::unique_ptr<t::Unit> & unit) {
  os << unit->name() << '(' << unit->value() << ')';
  return os;
}

#define MAKE(UNIT)                                                             \
  std::unique_ptr<Unit> Unit::UNIT(double value) {                             \
    return std::make_unique<t::UNIT>(value);                                   \
  }                                                                            \
  std::unique_ptr<Unit> Unit::UNIT(const std::unique_ptr<Unit> & unit) {       \
    return Convert<&UnitBase::To##UNIT>(unit);                                 \
  }

MAKE(Celsius)
MAKE(Fahrenheit)
MAKE(Kelvin)
MAKE(Reaumur)
MAKE(Delisle)
MAKE(Rankine)
MAKE(Romer)
MAKE(Newton)

}  // namespace t
