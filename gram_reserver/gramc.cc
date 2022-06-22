#include <iostream>
#include <memory>
#include <sstream>
#include <thread>

#include <cuda_runtime.h>

class Reserver {
public:
  virtual ~Reserver() = default;

  static std::unique_ptr<Reserver> ByFactor(const std::string & str);
};

class FactorReserver : public Reserver {
public:
  explicit FactorReserver(double factor) : factor_{factor} {
    std::size_t sizeCount = 1000 * 1000 * factor;
    std::cout << ">>> " << sizeCount << std::endl;
    cudaMalloc(&devPointer_, sizeCount);
    cudaMemset(devPointer_, -1, sizeCount);
  }

  ~FactorReserver() { cudaFree(devPointer_); }

private:
  double factor_;
  void * devPointer_;
};

std::unique_ptr<Reserver> Reserver::ByFactor(const std::string & str) {
  std::istringstream iss(str);
  double factor = 0;
  iss >> factor;
  return std::make_unique<FactorReserver>(factor);
}

int main(int argc, char * argv[]) {
  if (argc != 3) {
    std::cerr << "usage: " << argv[0] << " <factor> <sleep>" << std::endl;
    return EXIT_FAILURE;
  }

  const std::string factorArg{argv[1]};
  const std::string sleepArg{argv[2]};

  std::unique_ptr<Reserver> reserver = Reserver::ByFactor(factorArg);

  std::istringstream iss(sleepArg);
  std::size_t seconds = 0;
  iss >> seconds;
  std::this_thread::sleep_for(std::chrono::seconds(seconds));

  return EXIT_SUCCESS;
}
