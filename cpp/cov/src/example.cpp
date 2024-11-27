#include <iostream>

#include <converter.hpp>

void ShowNames() {
  auto celsius = t::Unit::Celsius(0);
  auto fahrenheit = t::Unit::Fahrenheit(0);
  auto kelvin = t::Unit::Kelvin(0);
  auto reaumur = t::Unit::Reaumur(0);
  auto delisle = t::Unit::Delisle(0);
  auto rankine = t::Unit::Rankine(0);
  auto romer = t::Unit::Romer(0);
  auto newton = t::Unit::Newton(0);

  std::cout << "Names" << std::endl
            << "\tcelsius: " << celsius->name() << std::endl
            << "\tfahrenheit: " << fahrenheit->name() << std::endl
            << "\tkelvin: " << kelvin->name() << std::endl
            << "\treaumur: " << reaumur->name() << std::endl
            << "\tdelisle: " << delisle->name() << std::endl
            << "\trankine: " << rankine->name() << std::endl
            << "\tromer: " << romer->name() << std::endl
            << "\tnewton: " << newton->name() << std::endl;
}

void ShowRepresentation() {
  auto celsius = t::Unit::Celsius(37);
  auto fahrenheit = t::Unit::Fahrenheit(37);
  auto kelvin = t::Unit::Kelvin(37);
  auto reaumur = t::Unit::Reaumur(37);
  auto delisle = t::Unit::Delisle(37);
  auto rankine = t::Unit::Rankine(37);
  auto romer = t::Unit::Romer(37);
  auto newton = t::Unit::Newton(37);

  std::cout << "Representation for 37" << std::endl
            << "\tcelsius = " << celsius << std::endl
            << "\tfahrenheit: " << fahrenheit << std::endl
            << "\tkelvin: " << kelvin << std::endl
            << "\treaumur: " << reaumur << std::endl
            << "\tdelisle: " << delisle << std::endl
            << "\trankine: " << rankine << std::endl
            << "\tromer: " << romer << std::endl
            << "\tnewton: " << newton << std::endl;
}

void ShowConversions() {
  auto celsius = t::Unit::Celsius(0);
  auto fahrenheit = t::Unit::Fahrenheit(celsius);
  auto kelvin = t::Unit::Kelvin(fahrenheit);
  auto reaumur = t::Unit::Reaumur(kelvin);
  auto delisle = t::Unit::Delisle(reaumur);
  auto rankine = t::Unit::Rankine(delisle);
  auto romer = t::Unit::Romer(rankine);
  auto newton = t::Unit::Newton(romer);
  auto celsius_ = t::Unit::Celsius(newton);

  std::cout << "Conversions" << std::endl
            << '\t' << celsius << " to " << fahrenheit << std::endl
            << '\t' << fahrenheit << " to " << kelvin << std::endl
            << '\t' << kelvin << " to " << reaumur << std::endl
            << '\t' << reaumur << " to " << delisle << std::endl
            << '\t' << delisle << " to " << rankine << std::endl
            << '\t' << rankine << " to " << romer << std::endl
            << '\t' << romer << " to " << newton << std::endl
            << '\t' << newton << " to " << celsius_ << std::endl;
}

int main() {
  ShowNames();
  std::cout << std::endl;
  ShowRepresentation();
  std::cout << std::endl;
  ShowConversions();
  return 0;
}
