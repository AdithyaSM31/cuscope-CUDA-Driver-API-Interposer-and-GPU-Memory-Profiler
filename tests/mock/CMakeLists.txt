set(MOCK_DIR ${CMAKE_BINARY_DIR}/mock)

# The mock driver: build/mock/driver/libcuda.so.1.1, SONAME libcuda.so.1.
add_library(mock_driver SHARED mock_driver.cpp)
target_include_directories(mock_driver PRIVATE ${CUDA_DRIVER_INCLUDE})
# Like the real driver, bind the mock's references to its own functions inside
# the library. Without this, cuGetProcAddress would return cuscope's preloaded
# exports of the same names instead of the mock's functions.
target_link_options(mock_driver PRIVATE -Wl,-Bsymbolic)
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

# A fake shared runtime named like the real one: build/mock/rt/libcudart.so.13.
add_library(fake_cudart SHARED fake_cudart.cpp)
target_link_libraries(fake_cudart PRIVATE dl)
# A tail call (jmp dlsym) would make the program, not this library, look like
# dlsym's caller; the real libcudart makes a normal call.
target_compile_options(fake_cudart PRIVATE -fno-optimize-sibling-calls)
set_target_properties(fake_cudart PROPERTIES
  OUTPUT_NAME cudart SUFFIX ".so.13"
  LIBRARY_OUTPUT_DIRECTORY ${MOCK_DIR}/rt)

add_executable(runtime_mimic runtime_mimic.cpp)
target_include_directories(runtime_mimic PRIVATE ${CUDA_DRIVER_INCLUDE} ${PROJECT_SOURCE_DIR}/include)
target_compile_definitions(runtime_mimic PRIVATE FAKE_CUDART_PATH="$<TARGET_FILE:fake_cudart>")
target_link_libraries(runtime_mimic PRIVATE dl Threads::Threads)
add_dependencies(runtime_mimic fake_cudart)
set_target_properties(runtime_mimic PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${MOCK_DIR})

# Linked with -lcuda against the mock; LD_LIBRARY_PATH picks the mode at run time.
add_executable(direct_mimic direct_mimic.cpp)
target_include_directories(direct_mimic PRIVATE ${CUDA_DRIVER_INCLUDE} ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(direct_mimic PRIVATE mock_driver dl)
set_target_properties(direct_mimic PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${MOCK_DIR}
  SKIP_BUILD_RPATH TRUE)

# Programs that use dlsym's caller-dependent handles (RTLD_NEXT, RTLD_DEFAULT
# from an RTLD_LOCAL plugin): cuscope's override must keep them working.
add_library(rtld_wrap SHARED rtld_wrap.c)
target_link_libraries(rtld_wrap PRIVATE dl)
add_library(rtld_dep SHARED rtld_dep.c)
add_library(rtld_plugin SHARED rtld_plugin.c)
target_link_libraries(rtld_plugin PRIVATE rtld_dep dl)
add_executable(rtld_app rtld_app.c)
target_link_libraries(rtld_app PRIVATE rtld_wrap dl)
target_compile_definitions(rtld_app PRIVATE RTLD_PLUGIN_PATH="$<TARGET_FILE:rtld_plugin>")
add_dependencies(rtld_app rtld_plugin)
set_target_properties(rtld_app PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${MOCK_DIR})

# dlsym lookups cuscope must leave alone: a non-libcuda library exporting a
# driver name, and a cuGetProcAddress probe before libcuda is loaded.
add_library(fakecu SHARED fakecu.c)
add_executable(dlsym_probe dlsym_probe.c)
target_link_libraries(dlsym_probe PRIVATE dl)
target_compile_definitions(dlsym_probe PRIVATE FAKECU_PATH="$<TARGET_FILE:fakecu>")
add_dependencies(dlsym_probe fakecu)
set_target_properties(dlsym_probe PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${MOCK_DIR})

add_test(NAME mock_suite
  COMMAND python3 -m pytest -p no:cacheprovider -q ${CMAKE_CURRENT_SOURCE_DIR})
set_tests_properties(mock_suite PROPERTIES
  LABELS mock
  ENVIRONMENT "CUSCOPE_BUILD=${CMAKE_BINARY_DIR};CUSCOPE_PRELOAD_FIRST=${CUSCOPE_PRELOAD_FIRST}")
