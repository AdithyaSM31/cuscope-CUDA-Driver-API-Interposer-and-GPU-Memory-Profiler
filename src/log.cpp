#include "log.hpp"

#include "config.hpp"

#include <fcntl.h>
#include <pthread.h>
#include <unistd.h>

#include <cstdarg>
#include <cstdio>

namespace cuscope {
namespace {

// Plain ints with constant initialisers: usable before cuscope's constructors run.
pthread_once_t g_log_once = PTHREAD_ONCE_INIT;
int g_log_fd = STDERR_FILENO;

void open_log() {
  const std::string& path = config().log_path;
  if (path.empty()) return;
  int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
  if (fd >= 0) g_log_fd = fd;
}

}  // namespace

void log_msg(const char* fmt, ...) {
  pthread_once(&g_log_once, open_log);

  constexpr char kPrefix[] = "[cuscope] ";
  char buf[1024];
  int n = std::snprintf(buf, sizeof(buf), "%s", kPrefix);

  va_list args;
  va_start(args, fmt);
  int body = std::vsnprintf(buf + n, sizeof(buf) - n - 1, fmt, args);
  va_end(args);
  if (body < 0) return;

  n += body;
  if (n > static_cast<int>(sizeof(buf)) - 2) n = sizeof(buf) - 2;  // truncated
  buf[n++] = '\n';

  ssize_t ignored = ::write(g_log_fd, buf, n);
  (void)ignored;
}

void log_write(const char* text, size_t length) {
  pthread_once(&g_log_once, open_log);
  while (length > 0) {
    ssize_t n = ::write(g_log_fd, text, length);
    if (n <= 0) return;
    text += n;
    length -= static_cast<size_t>(n);
  }
}

}  // namespace cuscope
