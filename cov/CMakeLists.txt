cmake_minimum_required(VERSION 3.21)

project(cov)

add_library(unit src/converter.cpp)

add_executable(converter-test src/converter-test.cpp)

add_definitions(-fprofile-arcs -ftest-coverage)
target_link_libraries(converter-test -fprofile-arcs)

target_include_directories(converter-test PUBLIC src $ENV{CONDA_PREFIX}/include)
target_link_directories(converter-test PUBLIC $ENV{CONDA_PREFIX}/lib)

target_link_libraries(converter-test gtest gtest_main)

target_include_directories(converter-test PUBLIC .)
target_link_libraries(converter-test unit)


set(OBJECT_DIR ${CMAKE_BINARY_DIR}/gcov})
# ${CMAKE_BINARY_DIR}/CMakeFiles/converter-test.dir/)
message("-- Object files will be output to: ${OBJECT_DIR}")

add_custom_target(init
    COMMAND ${CMAKE_MAKE_PROGRAM} clean
    COMMAND rm -f ${OBJECT_DIR}/*.gcno
    COMMAND rm -f ${OBJECT_DIR}/*.gcda
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

add_custom_target(gcov
    COMMAND mkdir -p gcov
    COMMAND ${CMAKE_MAKE_PROGRAM} converter-test
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

add_custom_command(TARGET gcov
    COMMAND echo "=================== GCOV ===================="
    COMMAND echo gcov -b ${CMAKE_SOURCE_DIR}/src/*.cpp -o ${OBJECT_DIR}
    COMMAND echo "-- Source diretory: ${CMAKE_SOURCE_DIR}/"
    COMMAND echo "-- Coverage files have been output to ${CMAKE_BINARY_DIR}/gcov"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/gcov)

add_dependencies(gcov converter-test)

add_custom_target(lcov COMMAND mkdir -p lcov)

add_custom_command(TARGET lcov
    COMMAND echo "=================== LCOV ===================="
    COMMAND echo "-- Passing lcov tool under code coverage"
    COMMAND lcov --capture --directory ../ --output-file lcov/main_coverage.info
    COMMAND echo "-- Generating HTML output files"
    COMMAND genhtml lcov/main_coverage.info --output-directory lcov)

add_dependencies(lcov gcov)

enable_testing()

include(GoogleTest)
gtest_discover_tests(converter-test)
