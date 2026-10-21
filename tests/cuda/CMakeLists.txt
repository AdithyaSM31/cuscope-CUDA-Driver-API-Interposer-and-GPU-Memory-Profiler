set(CMAKE_CUDA_ARCHITECTURES 86)

# Adds a CUDA test program built with -g -lineinfo so stacks resolve to file:line.
# RUNTIME is Static (nvcc default) or Shared (-cudart shared).
function(cuscope_cuda_test name source runtime)
  add_executable(${name} ${source})
  set_target_properties(${name} PROPERTIES CUDA_RUNTIME_LIBRARY ${runtime})
  target_compile_options(${name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-g -lineinfo>)
endfunction()

cuscope_cuda_test(hello_static hello.cu Static)
cuscope_cuda_test(hello_shared hello.cu Shared)
