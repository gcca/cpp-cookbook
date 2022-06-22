cmake_minimum_required(VERSION 3.21)
project(gramc LANGUAGES CUDA CXX)
add_executable(gramc gramc.cc)
set_source_files_properties(gramc.cc PROPERTIES LANGUAGE CUDA)
