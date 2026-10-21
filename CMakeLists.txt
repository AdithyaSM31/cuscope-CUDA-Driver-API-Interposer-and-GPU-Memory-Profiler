cmake_minimum_required(VERSION 3.28)
project(cuscope VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(CUSCOPE_BUILD_TESTS "Build unit and mock tests" ON)

# GPU test programs need nvcc. Non-interactive WSL shells often lack the toolkit
# on PATH, so fall back to the pinned install location.
include(CheckLanguage)
if(NOT CMAKE_CUDA_COMPILER AND NOT DEFINED ENV{CUDACXX} AND EXISTS /usr/local/cuda-13.0/bin/nvcc)
  set(CMAKE_CUDA_COMPILER /usr/local/cuda-13.0/bin/nvcc)
endif()
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  set(_cuscope_gpu_default ON)
else()
  set(_cuscope_gpu_default OFF)
endif()
option(CUSCOPE_GPU_TESTS "Build CUDA test programs (needs nvcc)" ${_cuscope_gpu_default})

# cuscope compiles against cuda.h only and never links libcuda.
find_path(CUDA_DRIVER_INCLUDE cuda.h
  HINTS /usr/local/cuda-13.0/include $ENV{CUDA_HOME}/include
  REQUIRED)
find_package(Threads REQUIRED)

# Testable core: everything except the code that hooks into the dynamic linker.
add_library(cuscope_core OBJECT
  src/log.cpp
  src/config.cpp
  src/stats.cpp
  src/census.cpp
  src/hook_names.cpp
  src/summary.cpp
  src/trace_writer.cpp
  src/report.cpp
  src/tracker.cpp
  src/stacks.cpp
  src/recorder.cpp)
target_include_directories(cuscope_core PUBLIC src include ${CUDA_DRIVER_INCLUDE})
target_compile_definitions(cuscope_core PUBLIC CUSCOPE_VERSION="${PROJECT_VERSION}")
target_compile_options(cuscope_core PRIVATE -Wall -Wextra -fvisibility=hidden -fvisibility-inlines-hidden)

# The preload library: core objects plus the interception layer.
set(CUSCOPE_SOURCES
  $<TARGET_OBJECTS:cuscope_core>
  src/runtime.cpp
  src/loader.cpp
  src/intercept.cpp
  src/hook_table.cpp
  src/gputimer.cpp
  src/wrappers/alloc.cpp
  src/wrappers/memcpy.cpp
  src/wrappers/launch.cpp
  src/wrappers/teardown.cpp
  src/wrappers/names.cpp)
add_library(cuscope SHARED ${CUSCOPE_SOURCES})
target_include_directories(cuscope PRIVATE src include ${CUDA_DRIVER_INCLUDE})
target_compile_definitions(cuscope PRIVATE CUSCOPE_VERSION="${PROJECT_VERSION}")
target_compile_options(cuscope PRIVATE -Wall -Wextra -fvisibility=hidden -fvisibility-inlines-hidden)
target_link_libraries(cuscope PRIVATE dl Threads::Threads)
target_link_options(cuscope PRIVATE -Wl,--no-undefined)
set_target_properties(cuscope PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

if(CUSCOPE_BUILD_TESTS)
  enable_testing()
  enable_language(ASM)  # the mock shim's generated trampolines
  add_subdirectory(tests/unit)
  add_subdirectory(tests/mock)
endif()

if(CUSCOPE_BUILD_TESTS)
  add_test(NAME cli_suite
    COMMAND python3 -m pytest -p no:cacheprovider -q ${PROJECT_SOURCE_DIR}/tests/cli)
  set_tests_properties(cli_suite PROPERTIES
    LABELS cli
    ENVIRONMENT "CUSCOPE_BUILD=${CMAKE_BINARY_DIR}")
endif()

if(CUSCOPE_GPU_TESTS)
  enable_language(CUDA)
  add_subdirectory(tests/cuda)
endif()
