#include "config.hpp"

#include <gtest/gtest.h>

#include <map>
#include <memory>
#include <string>

using namespace cuscope;

namespace {

// Fake getenv over a fixed map, so tests never depend on the real environment.
GetenvFn env(std::map<std::string, std::string> vars) {
  auto shared = std::make_shared<std::map<std::string, std::string>>(std::move(vars));
  return [shared](const char* key) -> const char* {
    auto it = shared->find(key);
    return it == shared->end() ? nullptr : it->second.c_str();
  };
}

}  // namespace

TEST(Config, Defaults) {
  auto c = parse_config(env({}));
  EXPECT_EQ(c.out, "cuscope-%p.json");
  EXPECT_EQ(c.mode, Config::Mode::Full);
  EXPECT_EQ(c.stack_depth, 16);
  EXPECT_EQ(c.sample_every, 1u);
  EXPECT_TRUE(c.log_path.empty());
  EXPECT_TRUE(c.census_path.empty());
  EXPECT_TRUE(c.warnings.empty());
}

TEST(Config, ReadsAll) {
  auto c = parse_config(env({{"CUSCOPE_MODE", "alloc"},
                             {"CUSCOPE_STACK_DEPTH", "8"},
                             {"CUSCOPE_SAMPLE_EVERY", "10"},
                             {"CUSCOPE_OUT", "/tmp/t.json"},
                             {"CUSCOPE_LOG", "/tmp/l.log"},
                             {"CUSCOPE_CENSUS", "/tmp/c.tsv"}}));
  EXPECT_EQ(c.mode, Config::Mode::Alloc);
  EXPECT_EQ(c.stack_depth, 8);
  EXPECT_EQ(c.sample_every, 10u);
  EXPECT_EQ(c.out, "/tmp/t.json");
  EXPECT_EQ(c.log_path, "/tmp/l.log");
  EXPECT_EQ(c.census_path, "/tmp/c.tsv");
  EXPECT_TRUE(c.warnings.empty());
}

// Review Focus 5: bad values never break the program; they fall back and warn once each.
TEST(Config, InvalidValuesFallBack) {
  auto c = parse_config(env({{"CUSCOPE_MODE", "bogus"},
                             {"CUSCOPE_STACK_DEPTH", "abc"},
                             {"CUSCOPE_SAMPLE_EVERY", "0"}}));
  EXPECT_EQ(c.mode, Config::Mode::Full);
  EXPECT_EQ(c.stack_depth, 16);
  EXPECT_EQ(c.sample_every, 1u);
  EXPECT_EQ(c.warnings.size(), 3u);

  EXPECT_EQ(parse_config(env({{"CUSCOPE_STACK_DEPTH", "1000"}})).stack_depth, 64);
  EXPECT_EQ(parse_config(env({{"CUSCOPE_STACK_DEPTH", "0"}})).stack_depth, 1);
  EXPECT_EQ(parse_config(env({{"CUSCOPE_SAMPLE_EVERY", "-5"}})).sample_every, 1u);
}

TEST(Config, MemLimitSizes) {
  auto limit = [](const char* v) { return parse_config(env({{"CUSCOPE_MEM_LIMIT", v}})).mem_limit_bytes; };
  EXPECT_EQ(parse_config(env({})).mem_limit_bytes, 0u);
  EXPECT_EQ(limit("268435456"), 268435456u);
  EXPECT_EQ(limit("512M"), 512ull << 20);
  EXPECT_EQ(limit("2G"), 2ull << 30);
  EXPECT_EQ(limit("2g"), 2ull << 30);
  EXPECT_EQ(limit("1.5GiB"), 3ull << 29);
  EXPECT_EQ(limit("64KB"), 64ull << 10);
  EXPECT_EQ(limit("1T"), 1ull << 40);
}

TEST(Config, MemLimitRejectsJunk) {
  for (const char* v : {"", "0", "-1G", "G", "2X", "1.5", "two"}) {
    Config c = parse_config(env({{"CUSCOPE_MEM_LIMIT", v}}));
    EXPECT_EQ(c.mem_limit_bytes, 0u) << v;
    EXPECT_EQ(c.warnings.size(), 1u) << v;
  }
}

TEST(Config, FlushTimeout) {
  EXPECT_EQ(parse_config(env({})).flush_timeout_ms, 2000u);
  EXPECT_EQ(parse_config(env({{"CUSCOPE_FLUSH_TIMEOUT_MS", "250"}})).flush_timeout_ms, 250u);
  EXPECT_EQ(parse_config(env({{"CUSCOPE_FLUSH_TIMEOUT_MS", "0"}})).flush_timeout_ms, 0u);
  auto bad = parse_config(env({{"CUSCOPE_FLUSH_TIMEOUT_MS", "-1"}}));
  EXPECT_EQ(bad.flush_timeout_ms, 2000u);
  EXPECT_EQ(bad.warnings.size(), 1u);
}

TEST(Config, EmptyOutKeepsDefault) {
  EXPECT_EQ(parse_config(env({{"CUSCOPE_OUT", ""}})).out, "cuscope-%p.json");
}

TEST(Config, ExpandsPid) {
  EXPECT_EQ(expand_out_path("a-%p.json", 42), "a-42.json");
  EXPECT_EQ(expand_out_path("plain.json", 42), "plain.json");
  EXPECT_EQ(expand_out_path("%p/%p.json", 7), "7/7.json");
}

TEST(Config, FixedPathBelongsToTheRootProcess) {
  // Without %p every process would write the same file; only the process
  // that started the run gets the exact path, the others add their pid.
  EXPECT_EQ(expand_out_path("trace.json", 42, /*root=*/true), "trace.json");
  EXPECT_EQ(expand_out_path("trace.json", 42, /*root=*/false), "trace.42.json");
  EXPECT_EQ(expand_out_path("out/trace", 42, /*root=*/false), "out/trace.42");
  EXPECT_EQ(expand_out_path("v1.2/trace", 42, /*root=*/false), "v1.2/trace.42");
  EXPECT_EQ(expand_out_path("t-%p.json", 42, /*root=*/false), "t-42.json");
}
