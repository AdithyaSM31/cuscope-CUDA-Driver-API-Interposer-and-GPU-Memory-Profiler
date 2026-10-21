set(CMAKE_CUDA_ARCHITECTURES 86)
find_package(CUDAToolkit REQUIRED)

# Adds a CUDA test program built with -g -lineinfo so stacks resolve to file:line.
# RUNTIME is Static (nvcc default) or Shared (-cudart shared).
function(cuscope_cuda_test name source runtime)
  add_executable(${name} ${source})
  set_target_properties(${name} PROPERTIES CUDA_RUNTIME_LIBRARY ${runtime})
  target_compile_options(${name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-g -lineinfo>)
endfunction()

cuscope_cuda_test(hello_static hello.cu Static)
cuscope_cuda_test(hello_shared hello.cu Shared)
cuscope_cuda_test(leak leak.cu Static)
cuscope_cuda_test(double_free double_free.cu Static)
cuscope_cuda_test(async_pool async_pool.cu Static)
cuscope_cuda_test(multistream multistream.cu Static)
cuscope_cuda_test(bandwidth bandwidth.cu Static)
cuscope_cuda_test(graph_capture graph_capture.cu Static)
cuscope_cuda_test(device_reset device_reset.cu Static)
cuscope_cuda_test(capture_race capture_race.cu Static)
cuscope_cuda_test(exit_busy exit_busy.cu Static)
cuscope_cuda_test(fork_after_cuda fork_after_cuda.cu Static)

# Host-only C++ program using the runtime API from eight threads.
add_executable(multithread multithread.cpp)
target_compile_options(multithread PRIVATE -g)
target_link_libraries(multithread PRIVATE CUDA::cudart_static Threads::Threads)

add_test(NAME check_exports
  COMMAND python3 ${PROJECT_SOURCE_DIR}/tools/check_exports.py ${PROJECT_SOURCE_DIR}/src/hooks.def)
set_tests_properties(check_exports PROPERTIES LABELS gpu)

add_test(NAME gpu_suite
  COMMAND python3 -m pytest -p no:cacheprovider -q ${PROJECT_SOURCE_DIR}/tests/gpu)
set_tests_properties(gpu_suite PROPERTIES
  LABELS gpu
  ENVIRONMENT "CUSCOPE_BUILD=${CMAKE_BINARY_DIR}")
