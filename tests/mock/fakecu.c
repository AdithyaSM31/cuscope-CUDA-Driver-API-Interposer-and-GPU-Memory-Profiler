/* A library that is not libcuda but happens to export a driver function name.
 * cuscope must not hand its wrapper to a program looking it up here. */
typedef unsigned long long CUdeviceptr_fake;

int cuMemAlloc_v2(CUdeviceptr_fake* ptr, unsigned long long bytes) {
  (void)bytes;
  *ptr = 0;
  return 999;
}
