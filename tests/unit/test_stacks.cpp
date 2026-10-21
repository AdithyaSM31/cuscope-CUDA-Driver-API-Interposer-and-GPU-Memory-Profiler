#include "stacks.hpp"

#include <gtest/gtest.h>

#include <string>

using namespace cuscope;

// Not static and not inlined, so each is a real frame that can be named. The
// empty asm keeps the call out of tail position: `return capture_stack(0)`
// may compile to a jump, and then this function's frame is not on the stack.
#define KEEP_FRAME(id) asm volatile("" : "+r"(id))
extern "C" __attribute__((noinline)) uint32_t stack_marker_function() {
  uint32_t id = capture_stack(0);
  KEEP_FRAME(id);
  return id;
}
extern "C" __attribute__((noinline)) uint32_t other_call_site() {
  uint32_t id = capture_stack(0);
  KEEP_FRAME(id);
  return id;
}
extern "C" __attribute__((noinline)) uint32_t shallow_capture() {
  uint32_t id = capture_stack(0, 4);
  KEEP_FRAME(id);
  return id;
}

TEST(Stacks, SameSiteSameId) {
  // volatile: an unrolled loop would make two different call sites.
  uint32_t ids[2];
  for (volatile int i = 0; i < 2; i = i + 1) ids[i] = stack_marker_function();
  EXPECT_NE(ids[0], 0u);
  EXPECT_EQ(ids[0], ids[1]);
}

TEST(Stacks, DifferentSitesDifferentIds) {
  EXPECT_NE(stack_marker_function(), other_call_site());
}

TEST(Stacks, DepthRespected) {
  uint32_t id = shallow_capture();
  auto stacks = symbolize_all();
  ASSERT_TRUE(stacks.count(id));
  EXPECT_GE(stacks[id].size(), 1u);
  EXPECT_LE(stacks[id].size(), 4u);
}

TEST(Stacks, SymbolizeHasModuleAndSymbol) {
  uint32_t id = stack_marker_function();
  auto stacks = symbolize_all();
  ASSERT_TRUE(stacks.count(id));
  const auto& frames = stacks[id];
  ASSERT_FALSE(frames.empty());
  EXPECT_NE(frames[0].symbol.find("stack_marker_function"), std::string::npos) << frames[0].symbol;
  EXPECT_NE(frames[0].module.find("cuscope_unit_tests"), std::string::npos) << frames[0].module;
  EXPECT_GT(frames[0].offset, 0u);
}

TEST(Stacks, Demangle) {
  EXPECT_EQ(demangle("_Z11spin_kernelx"), "spin_kernel(long long)");
  EXPECT_EQ(demangle("main"), "main");
  EXPECT_EQ(demangle(""), "");
}
