#include <iostream>

class Fullname {
 public:
  Fullname(std::string first, std::string second)
      : first_{std::move(first)}, second_{std::move(second)} {}

  void Print() const noexcept {
    std::cout << second_ << ", " << first_ << std::endl;
  }

 private:
  std::string first_;
  std::string second_;
};

template <class T>
class Repeated : public T {
 public:
  constexpr explicit Repeated(T const &t) : T{t} {}

  constexpr void Print(std::size_t n) const noexcept {
    while (n-- > 0) {
      this->T::Print();
    }
  }
};

int main() {
  Fullname name("Uno", "Dos");
  Repeated(name).Print(10);
  return 0;
}
