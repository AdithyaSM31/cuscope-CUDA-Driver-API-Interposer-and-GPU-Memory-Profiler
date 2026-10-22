/* dlsym lookups that cuscope must answer exactly as glibc would.
 *
 *   dlsym_probe foreign  - cuMemAlloc_v2 from a library that is not libcuda
 *   dlsym_probe probe    - cuGetProcAddress via RTLD_DEFAULT, libcuda not loaded
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>

static int libcuda_mapped(void) {
  FILE* maps = fopen("/proc/self/maps", "r");
  char line[512];
  int found = 0;
  while (maps != NULL && fgets(line, sizeof line, maps) != NULL) {
    if (strstr(line, "/libcuda.so") != NULL) found = 1;
  }
  if (maps != NULL) fclose(maps);
  return found;
}

int main(int argc, char** argv) {
  if (argc == 2 && strcmp(argv[1], "foreign") == 0) {
    void* lib = dlopen(FAKECU_PATH, RTLD_NOW | RTLD_LOCAL);
    if (lib == NULL) return 2;
    int (*alloc)(unsigned long long*, unsigned long long) =
        (int (*)(unsigned long long*, unsigned long long))dlsym(lib, "cuMemAlloc_v2");
    unsigned long long ptr = 0;
    printf("foreign cuMemAlloc_v2: %d\n", alloc != NULL ? alloc(&ptr, 16) : -1);
    return 0;
  }
  if (argc == 2 && strcmp(argv[1], "probe") == 0) {
    void* v1 = dlsym(RTLD_DEFAULT, "cuGetProcAddress");
    void* v2 = dlsym(RTLD_DEFAULT, "cuGetProcAddress_v2");
    printf("probe: %s %s, libcuda loaded: %s\n", v1 ? "found" : "null", v2 ? "found" : "null",
           libcuda_mapped() ? "yes" : "no");
    return 0;
  }
  return 2;
}
