/* An LD_PRELOAD-style interposer: defines puts and forwards to the next
 * definition with dlsym(RTLD_NEXT, "puts"). glibc resolves RTLD_NEXT from the
 * caller of dlsym, so if cuscope's dlsym override hides the real caller this
 * library gets its own puts back and recurses. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <unistd.h>

static int depth;

int puts(const char* s) {
  if (++depth > 1) _exit(99); /* RTLD_NEXT returned this very function */
  int (*next)(const char*) = (int (*)(const char*))dlsym(RTLD_NEXT, "puts");
  int r = next ? next(s) : -1;
  --depth;
  return r;
}
