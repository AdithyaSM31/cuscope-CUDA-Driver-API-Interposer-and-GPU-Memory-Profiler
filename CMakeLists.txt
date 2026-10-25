cmake_minimum_required(VERSION 3.28)
project(cuscope VERSION 0.2.0 LANGUAGES C CXX ASM)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(CUSCOPE_BUILD_TESTS "Build unit and mock tests" ON)
set(CUSCOPE_SANITIZE "" CACHE STRING "Build C and C++ code with a sanitizer: address or thread")

# Sanitizer builds instrument cuscope, the mock driver and the test programs
# (not CUDA device code). A preloaded instrumented library needs the sanitizer
# runtime ahead of it in LD_PRELOAD; the tests read its path from
# CUSCOPE_PRELOAD_FIRST.
set(CUSCOPE_PRELOAD_FIRST "")
if(CUSCOPE_SANITIZE)
  if(NOT CUSCOPE_SANITIZE MATCHES "^(address|thread)$")
    message(FATAL_ERROR "CUSCOPE_SANITIZE must be address or thread, not '${CUSCOPE_SANITIZE}'")
  endif()
  add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:-fsanitize=${CUSCOPE_SANITIZE}>
                      $<$<COMPILE_LANGUAGE:C,CXX>:-fno-omit-frame-pointer>)
  add_link_options($<$<LINK_LANGUAGE:C,CXX>:-fsanitize=${CUSCOPE_SANITIZE}>)
  if(CUSCOPE_SANITIZE STREQUAL "address")
    set(_sanitizer_runtime libasan.so)
  else()
    set(_sanitizer_runtime libtsan.so)
  endif()
  execute_process(COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=${_sanitizer_runtime}
                  OUTPUT_VARIABLE CUSCOPE_PRELOAD_FIRST OUTPUT_STRIP_TRAILING_WHITESPACE)
  set(CUSCOPE_GPU_TESTS OFF CACHE BOOL "" FORCE)  # nvcc-built programs are not instrumented
endif()

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
  src/limits.cpp
  src/sched_core.cpp
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
  src/dlsym_trampoline.S
  src/hook_table.cpp
  src/gputimer.cpp
  src/wrappers/alloc.cpp
  src/wrappers/meminfo.cpp
  src/sched_client.cpp
  src/wrappers/memcpy.cpp
  src/wrappers/launch.cpp
  src/wrappers/teardown.cpp
  src/wrappers/names.cpp)
add_library(cuscope SHARED ${CUSCOPE_SOURCES})
target_include_directories(cuscope PRIVATE src include ${CUDA_DRIVER_INCLUDE})
target_compile_definitions(cuscope PRIVATE CUSCOPE_VERSION="${PROJECT_VERSION}")
target_compile_options(cuscope PRIVATE -Wall -Wextra -fvisibility=hidden -fvisibility-inlines-hidden)
target_link_libraries(cuscope PRIVATE dl Threads::Threads)
target_link_options(cuscope PRIVATE -Wl,--no-undefined -Wl,--version-script=${PROJECT_SOURCE_DIR}/src/exports.map)
set_target_properties(cuscope PROPERTIES LINK_DEPENDS ${PROJECT_SOURCE_DIR}/src/exports.map)
# The C++ runtime is linked in and kept private (--exclude-libs), so the
# library works in a process that loads a different libstdc++ (a conda
# environment, say) and never exports its copy to the program. Sanitizer
# builds keep the shared runtime the sanitizer expects.
if(NOT CUSCOPE_SANITIZE)
  target_link_options(cuscope PRIVATE -static-libstdc++ -static-libgcc -Wl,--exclude-libs,ALL)
endif()
set_target_properties(cuscope PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# The compute-sharing daemon. It shares only the token math with the library
# and loads no CUDA; `cuscope sched` finds it next to libcuscope.so.
add_executable(cuscope-sched tools/cuscope_sched.cpp src/sched_core.cpp)
target_include_directories(cuscope-sched PRIVATE src)
target_compile_options(cuscope-sched PRIVATE -Wall -Wextra)
target_link_libraries(cuscope-sched PRIVATE rt)
set_target_properties(cuscope-sched PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

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

  # libcuscope.so is loaded into programs that may bring an older libstdc++
  # of their own (conda environments): it must not depend on the system one.
  if(NOT CUSCOPE_SANITIZE)
    add_test(NAME no_cxx_runtime_dependency
      COMMAND sh -c "! readelf -d '$<TARGET_FILE:cuscope>' | grep -E 'NEEDED.*(libstdc[+][+]|libgcc_s)'")
    set_tests_properties(no_cxx_runtime_dependency PROPERTIES LABELS cli)
  endif()
endif()

if(CUSCOPE_GPU_TESTS)
  enable_language(CUDA)
  add_subdirectory(tests/cuda)
  add_subdirectory(bench)
endif()
