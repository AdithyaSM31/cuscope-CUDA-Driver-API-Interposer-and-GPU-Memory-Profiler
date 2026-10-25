include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(googletest json)

add_executable(cuscope_unit_tests
  test_config.cpp
  test_limits.cpp
  test_recorder.cpp
  test_report.cpp
  test_sched_core.cpp
  test_ring.cpp
  test_stacks.cpp
  test_summary.cpp
  test_tracker.cpp
  test_trace_writer.cpp)
target_link_libraries(cuscope_unit_tests PRIVATE cuscope_core GTest::gtest_main nlohmann_json::nlohmann_json
                                                 Threads::Threads dl)
# -rdynamic: dladdr can only name functions in the executable's dynamic symbol table.
set_target_properties(cuscope_unit_tests PROPERTIES ENABLE_EXPORTS ON)

include(GoogleTest)
# PRE_TEST: list the tests when ctest runs, so building never executes the
# binary (a sanitizer build may not run on every machine that compiles it).
gtest_discover_tests(cuscope_unit_tests PROPERTIES LABELS unit DISCOVERY_MODE PRE_TEST)
