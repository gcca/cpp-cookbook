cmake_minimum_required(VERSION 3.21)

project(kol C CXX)

find_package(Threads REQUIRED)
find_package(Protobuf REQUIRED)
find_package(gRPC REQUIRED)

get_filename_component(k_proto kol.proto ABSOLUTE)
get_filename_component(k_proto_path "${k_proto}" PATH)

set(k_proto_srcs "${CMAKE_CURRENT_BINARY_DIR}/kol.pb.cc")
set(k_proto_hdrs "${CMAKE_CURRENT_BINARY_DIR}/kol.pb.h")
set(k_grpc_srcs "${CMAKE_CURRENT_BINARY_DIR}/kol.grpc.pb.cc")
set(k_grpc_hdrs "${CMAKE_CURRENT_BINARY_DIR}/kol.grpc.pb.h")
add_custom_command(
  OUTPUT "${k_proto_srcs}" "${k_proto_hdrs}" "${k_grpc_srcs}" "${k_grpc_hdrs}"
  COMMAND $<TARGET_FILE:protobuf::protoc>
  ARGS --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
    --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
    -I "${k_proto_path}"
    --plugin=protoc-gen-grpc=$<TARGET_FILE:gRPC::grpc_cpp_plugin>
    "${k_proto}"
  DEPENDS "${k_proto}")

add_definitions(-DNDEBUG)
include_directories("${CMAKE_CURRENT_BINARY_DIR}")

add_library(k_grpc_proto
  ${k_grpc_srcs} ${k_grpc_hdrs}
  ${k_proto_srcs} ${k_proto_hdrs})
target_link_libraries(k_grpc_proto
  gRPC::grpc++_reflection gRPC::grpc++ protobuf::libprotobuf)

foreach(target server client)
  add_executable(${target} ${target}.cc)
  target_link_libraries(${target} k_grpc_proto)
endforeach()
