#include "census.hpp"

#include "config.hpp"
#include "log.hpp"

#include <fcntl.h>
#include <pthread.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>

namespace cuscope {
namespace {

// Constant-initialised, so usable from a dlsym call that arrives before constructors.
pthread_once_t g_census_once = PTHREAD_ONCE_INIT;
int g_census_fd = -1;

void open_census() {
  const std::string& path = config().census_path;
  if (path.empty()) return;
  g_census_fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
  if (g_census_fd < 0) log_msg("cannot open CUSCOPE_CENSUS=%s, census disabled", path.c_str());
}

const char* basename_of(const char* path) {
  if (path == nullptr) return "?";
  const char* slash = std::strrchr(path, '/');
  return slash ? slash + 1 : path;
}

// One write(2) per line, so lines from different threads never interleave.
void write_line(const char* line, int len) {
  if (len <= 0) return;
  if (len > 1023) len = 1023;
  ssize_t ignored = ::write(g_census_fd, line, len);
  (void)ignored;
}

}  // namespace

bool census_enabled() {
  pthread_once(&g_census_once, open_census);
  return g_census_fd >= 0;
}

void census_dlsym(const char* symbol, const char* caller_module, const char* action) {
  if (!census_enabled()) return;
  char line[1024];
  int n = std::snprintf(line, sizeof(line), "dlsym\t%s\t%s\t%s\n", symbol ? symbol : "(null)",
                        basename_of(caller_module), action);
  write_line(line, n);
}

void census_gpa(const char* symbol, int cuda_version, uint64_t flags, int result, int status,
                const char* action) {
  if (!census_enabled()) return;
  char line[1024];
  int n = std::snprintf(line, sizeof(line), "gpa\t%s\t%d\t%llu\t%d\t%d\t%s\n",
                        symbol ? symbol : "(null)", cuda_version,
                        static_cast<unsigned long long>(flags), result, status, action);
  write_line(line, n);
}

}  // namespace cuscope
