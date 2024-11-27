#include <memory>

namespace t {

class Unit {
  friend std::ostream & operator<<(std::ostream & os,
                                   const std::unique_ptr<t::Unit> &);

public:
  virtual double value() const noexcept = 0;
  virtual const std::string & name() const noexcept = 0;

  static std::unique_ptr<Unit> Celsius(double);
  static std::unique_ptr<Unit> Fahrenheit(double);
  static std::unique_ptr<Unit> Kelvin(double);
  static std::unique_ptr<Unit> Reaumur(double);
  static std::unique_ptr<Unit> Delisle(double);
  static std::unique_ptr<Unit> Rankine(double);
  static std::unique_ptr<Unit> Romer(double);
  static std::unique_ptr<Unit> Newton(double);
  static std::unique_ptr<Unit> Celsius(const std::unique_ptr<Unit> &);
  static std::unique_ptr<Unit> Fahrenheit(const std::unique_ptr<Unit> &);
  static std::unique_ptr<Unit> Kelvin(const std::unique_ptr<Unit> &);
  static std::unique_ptr<Unit> Reaumur(const std::unique_ptr<Unit> &);
  static std::unique_ptr<Unit> Delisle(const std::unique_ptr<Unit> &);
  static std::unique_ptr<Unit> Rankine(const std::unique_ptr<Unit> &);
  static std::unique_ptr<Unit> Romer(const std::unique_ptr<Unit> &);
  static std::unique_ptr<Unit> Newton(const std::unique_ptr<Unit> &);
};

class Converter {
public:
};

}  // namespace t
