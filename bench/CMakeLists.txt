set(CMAKE_CUDA_ARCHITECTURES 86)

add_executable(micro micro.cu)
target_compile_options(micro PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O2>)
