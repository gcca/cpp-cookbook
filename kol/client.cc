#include <grpcpp/create_channel.h>

#include <iostream>

#include "kol.grpc.pb.h"

int main(int, char *[]) {
  std::cout << "CLIENT" << std::endl;

  std::shared_ptr<grpc::Channel> channel =
      grpc::CreateChannel("localhost:1234", grpc::InsecureChannelCredentials());
  std::unique_ptr<kol::Kol::Stub> stub = kol::Kol::NewStub(channel);

  kol::Person person;
  person.set_name("cristhian");
  person.set_age(22);

  grpc::ClientContext context;
  kol::Feature feature;
  grpc::Status status = stub->GetSigned(&context, person, &feature);

  if (!status.ok()) {
    std::cerr << "FAIL" << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "feature.sign = " << feature.sign() << std::endl;
  std::cout << "feature.person.name = " << feature.person().name() << std::endl;
  std::cout << "feature.person.age = " << feature.person().age() << std::endl;

  return EXIT_SUCCESS;
}
