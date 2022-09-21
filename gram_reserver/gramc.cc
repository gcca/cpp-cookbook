#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <thread>

#include <boost/program_options.hpp>

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

void Record(const std::string & factorArg, const std::string & sleepArg) {
  const std::string filename = "gramc_record";

  auto now = std::chrono::system_clock::now();
  std::time_t nowtime = std::chrono::system_clock::to_time_t(now);

  std::ofstream ofs(filename, ofs.binary | ofs.app);
  ofs << std::put_time(std::localtime(&nowtime), "%Y-%m-%d %X") << " "
      << factorArg << " " << sleepArg << std::endl;
}

void Wait(const std::string & sleepArg) {
  std::istringstream iss(sleepArg);
  std::size_t seconds = 0;
  iss >> seconds;
  std::this_thread::sleep_for(std::chrono::seconds(seconds));
}

int main(int argc, char * argv[]) {
  namespace po = boost::program_options;

  po::options_description desc("Allowed options");
  desc.add_options()("help", "produce help message")("compression",
                                                     po::value<int>(),
                                                     "set compression level");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cout << desc << "\n";
    return 1;
  }

  if (vm.count("compression")) {
    std::cout << "Compression level was set to " << vm["compression"].as<int>()
              << ".\n";
  } else {
    std::cout << "Compression level was not set.\n";
  }

  return 0;

  if (argc != 3) {
    std::cerr << "usage: " << argv[0] << " <factor> <sleep>" << std::endl;
    return EXIT_FAILURE;
  }

  const std::string factorArg{argv[1]};
  const std::string sleepArg{argv[2]};

  std::unique_ptr<Reserver> reserver = Reserver::ByFactor(factorArg);

  Record(factorArg, sleepArg);
  Wait(sleepArg);

  return EXIT_SUCCESS;
}
