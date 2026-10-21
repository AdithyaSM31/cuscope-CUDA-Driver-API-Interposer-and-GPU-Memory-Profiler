include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(cuscope_unit_tests
  test_config.cpp)
target_link_libraries(cuscope_unit_tests PRIVATE cuscope_core GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(cuscope_unit_tests PROPERTIES LABELS unit)
