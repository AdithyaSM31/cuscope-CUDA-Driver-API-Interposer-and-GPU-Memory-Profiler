set(MOCK_DIR ${CMAKE_BINARY_DIR}/mock)

# The mock driver: build/mock/driver/libcuda.so.1.1, SONAME libcuda.so.1.
add_library(mock_driver SHARED mock_driver.cpp)
target_include_directories(mock_driver PRIVATE ${CUDA_DRIVER_INCLUDE})
set_target_properties(mock_driver PROPERTIES
  OUTPUT_NAME cuda VERSION 1.1 SOVERSION 1
  LIBRARY_OUTPUT_DIRECTORY ${MOCK_DIR}/driver)

# "direct" mode: libcuda.so.1 is the mock driver itself.
add_custom_command(TARGET mock_driver POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory ${MOCK_DIR}/direct
  COMMAND ${CMAKE_COMMAND} -E create_symlink $<TARGET_FILE:mock_driver> ${MOCK_DIR}/direct/libcuda.so.1)

# "shim" mode: libcuda.so.1 is a WSL-style loader that forwards to the driver.
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/shim_stubs.s
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/gen_shim.py $<TARGET_FILE:mock_driver>
          ${CMAKE_CURRENT_BINARY_DIR}/shim_stubs.s
  DEPENDS mock_driver ${CMAKE_CURRENT_SOURCE_DIR}/gen_shim.py)
add_library(mock_shim SHARED mock_shim.cpp ${CMAKE_CURRENT_BINARY_DIR}/shim_stubs.s)
target_compile_definitions(mock_shim PRIVATE MOCK_DRIVER_PATH="$<TARGET_FILE:mock_driver>")
target_link_libraries(mock_shim PRIVATE dl)
set_target_properties(mock_shim PROPERTIES
  OUTPUT_NAME cuda SUFFIX ".so.1"
  LIBRARY_OUTPUT_DIRECTORY ${MOCK_DIR}/shim)

add_executable(runtime_mimic runtime_mimic.cpp)
target_include_directories(runtime_mimic PRIVATE ${CUDA_DRIVER_INCLUDE} ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(runtime_mimic PRIVATE dl)
set_target_properties(runtime_mimic PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${MOCK_DIR})

# Linked with -lcuda against the mock; LD_LIBRARY_PATH picks the mode at run time.
add_executable(direct_mimic direct_mimic.cpp)
target_include_directories(direct_mimic PRIVATE ${CUDA_DRIVER_INCLUDE} ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(direct_mimic PRIVATE mock_driver dl)
set_target_properties(direct_mimic PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${MOCK_DIR}
  SKIP_BUILD_RPATH TRUE)

add_test(NAME mock_suite
  COMMAND python3 -m pytest -p no:cacheprovider -q ${CMAKE_CURRENT_SOURCE_DIR})
set_tests_properties(mock_suite PROPERTIES
  LABELS mock
  ENVIRONMENT "CUSCOPE_BUILD=${CMAKE_BINARY_DIR}")
