/* Linked against librtld_wrap.so (which wraps puts) and, for the second
 * check, loads a plugin with RTLD_LOCAL the way Python loads extensions. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>

int main(void) {
  puts("rtld_next: ok");

  void* plugin = dlopen(RTLD_PLUGIN_PATH, RTLD_NOW | RTLD_LOCAL);
  if (plugin == NULL) {
    printf("plugin: %s\n", dlerror());
    return 1;
  }
  int (*probe)(void) = (int (*)(void))dlsym(plugin, "rtld_plugin_probe");
  printf("rtld_default from a local plugin: %s\n", probe && probe() ? "found" : "missing");
  return 0;
}
