/* Loaded with RTLD_LOCAL; probes a symbol of its own dependency with
 * dlsym(RTLD_DEFAULT). glibc includes the calling object's local scope in that
 * search, so the lookup depends on dlsym seeing the real caller. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stddef.h>

extern int rtld_dep_value;

/* A direct reference keeps the dependency (the linker drops unused ones). */
int rtld_plugin_dep_value(void) { return rtld_dep_value; }

int rtld_plugin_probe(void) { return dlsym(RTLD_DEFAULT, "rtld_dep_value") != NULL; }
