#include <grpcpp/server_builder.h>

#include <iostream>

#include "kol.grpc.pb.h"

class KolServiceImpl : public kol::Kol::Service {
 public:
  grpc::Status GetSigned(grpc::ServerContext* context,
                         const kol::Person* person,
                         kol::Feature* feature) final {
    std::cout << "CALLED" << std::endl;
    feature->set_sign("SIGNED-245gh");
    feature->mutable_person()->CopyFrom(*person);
    return grpc::Status::OK;
  }
};

int main(int, char*[]) {
  std::cout << "SERVER" << std::endl;

  std::string serverAddress("0.0.0.0:1234");
  KolServiceImpl service;

  grpc::ServerBuilder builder;
  builder.AddListeningPort(serverAddress, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
  std::cout << "Server listening on " << serverAddress << std::endl;
  server->Wait();

  return EXIT_SUCCESS;
}
