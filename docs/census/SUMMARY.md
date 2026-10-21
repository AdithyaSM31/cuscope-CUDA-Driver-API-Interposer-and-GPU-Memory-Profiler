# Day 2 symbol census

Run on 2026-10-21: RTX 3060 Laptop GPU, Windows driver 617.14, WSL2 Ubuntu 22.04,
CUDA toolkit 13.0.88, PyTorch 2.14.0+cu130. Raw logs are in this folder
(`static.tsv`, `shared.tsv`, `pytorch.tsv`, `runs.json`), and `tools/census.py`
regenerates them:

```bash
python3 tools/census.py run && python3 tools/census.py check
```

## Gate result: PASS (9/9)

| Check | Result |
|---|---|
| `hello_static`, `hello_shared` and the PyTorch matmul exit 0 with stdout identical with and without cuscope | PASS ×3 |
| Static build: `cuMemAlloc` arrives through `cuGetProcAddress` | PASS |
| Static and shared builds: the runtime's `dlsym(cuGetProcAddress_v2)` gets cuscope's wrapper | PASS ×2 |
| All three: the WSL shim's own lookups pass through untouched (`caller-libcuda`) | PASS ×3 |

The `LD_PRELOAD` design works on WSL2, so the proxy-library fallback is not needed.

## Findings

1. **One lookup route for every build.** The static and shared builds look up the
   same 346 driver functions, all through `cuGetProcAddress`. That is the
   driver-level argument for the `static_vs_shared` test. PyTorch looks up 370,
   through three separate CUDA runtimes in one process: `libcudart.so.13`, plus
   copies statically linked into `libcublas.so.13` and `libcublasLt.so.13`. Each
   one fetches `cuGetProcAddress_v2` with its own `dlsym`, and cuscope wraps all
   three.

2. **The WSL shim makes 739 lookups that must pass through.** `libcuda.so.1`
   (the loader) calls `dlsym` 731 times to fill its table from the real driver.
   The real driver, `libcuda.so.1.1`, calls it 8 more times for the WSL GPU
   paravirtualisation layer (`D3DKMT*`, `dxg*`). The caller check passes every
   one through.

3. **Bug found and fixed during the census.** The first caller check matched any
   module whose name starts with `libcuda`. That includes `libcudart.so.13`, so
   every lookup from the shared runtime was passed through unrecorded:
   `hello_shared` showed 0 driver lookups. The check now requires `libcuda` to be
   followed by `.` or `_`. `test_libcudart_caller_is_not_libcuda` covers it.

4. **The runtime asks for each function at its own minimum version, not 13000.**
   Examples: `cuMemAlloc@3020`, `cuLaunchKernel@4000` (legacy stream) and
   `@7000` (per-thread), `cuMemAllocAsync@11020`, `cuLaunchKernelEx@11060`. The
   spec's pointer matching (§4.3) is what makes this safe: cuscope compares the
   returned pointer, not the version number.

5. **⚠ Signature hazard for Task 6: `cuCtxSynchronize`.** The runtime requests it
   at version 2000 and also at 13000. In CUDA 13.0, version 13000 returns
   `cuCtxSynchronize_v2(CUcontext)`, but `cuda.h` declares the base name as
   `cuCtxSynchronize(void)` with no macro to `_v2`. If Task 6 resolves every
   hook's expected pointer at `CUDA_VERSION`, as the plan currently says, the
   `(void)` wrapper would be matched against the `_v2` pointer and would call it
   with no context argument. **Fix:** each hook resolves its expected pointer at
   the version whose signature matches its `cuda.h` declaration (`hooks.def`
   gains a version column, default `CUDA_VERSION`, with `cuCtxSynchronize` at
   2000). Then a GPU test should prove, for every hook, that the pointer returned
   at that version is the same function as the `dlsym` of the export whose
   prototype the wrapper uses, checked against the real driver
   (`libcuda.so.1.1`) directly.

6. **Hook-list delta.**
   - Requested and already planned: the whole spec §5 list, plus `cuKernelGetFunction`
     (Task 12).
   - In the list but never requested by the runtime: `cuMemAllocHost` (the runtime
     uses `cuMemHostAlloc`) and `cuCtxDestroy` (the runtime uses the
     primary-context API). Keep both for programs that call the driver API directly.
   - Requested, not hooked, and in scope to decide:
     - `cuMemsetD8/D32/D2D8(Async)`: GPU work that would not be timed.
     - `cuMemcpy2D/3D/Peer(Async)` and `cuMemcpy(3D)BatchAsync`: copies that would
       not be tracked.
     - `cuLaunchCooperativeKernel`, `cuLaunchHostFunc`: launches that would not be
       timed.
     - `cuMemHostRegister`: pinned memory that is registered, not allocated; copy
       pinning is detected separately with `cuPointerGetAttribute`.
   - Out of scope, as the spec says: PyTorch also requests the VMM APIs
     (`cuMemCreate`, `cuMemMap`, `cuMemAddressReserve`, `cuMemRelease`) for
     expandable segments. Task 15 sets `expandable_segments:False`.
     `cuMemPool*` pool management is invisible; stream-ordered allocations are
     tracked per call.

7. **Per-thread-stream variants.** 67 functions are requested with the
   per-thread flag, including every hooked copy, launch, stream-ordered
   allocation and `cuStreamSynchronize`. Each hook needs both variant wrappers.

# Generated tables (python3 tools/census.py summary)

## Lookups per program

| Program | Driver functions looked up | dlsym(cuGetProcAddress*) wrapped for | Shim/driver lookups passed through |
|---|---|---|---|
| static | 346 | hello_static | 739 |
| shared | 346 | libcudart.so.13 | 739 |
| pytorch | 370 | libcublas.so.13, libcublasLt.so.13, libcudart.so.13 | 739 |

## Phase 1 hook list: requested versions and flags

| Function | static | shared | pytorch |
|---|---|---|---|
| `cuMemAlloc` | 3020 default | 3020 default | 3020 default |
| `cuMemAllocPitch` | 3020 default | 3020 default | 3020 default |
| `cuMemFree` | 3020 default | 3020 default | 3020 default |
| `cuMemAllocAsync` | 11020 default, 11020 per-thread | 11020 default, 11020 per-thread | 11020 default, 11020 per-thread |
| `cuMemAllocFromPoolAsync` | 11020 default, 11020 per-thread | 11020 default, 11020 per-thread | 11020 default, 11020 per-thread |
| `cuMemFreeAsync` | 11020 default, 11020 per-thread | 11020 default, 11020 per-thread | 11020 default, 11020 per-thread |
| `cuMemAllocManaged` | 6000 default | 6000 default | 6000 default |
| `cuMemAllocHost` | — | — | — |
| `cuMemHostAlloc` | 2020 default | 2020 default | 2020 default |
| `cuMemFreeHost` | 2000 default | 2000 default | 2000 default |
| `cuMemcpy` | 4000 default, 7000 per-thread | 4000 default, 7000 per-thread | 4000 default, 7000 per-thread |
| `cuMemcpyAsync` | 4000 default, 7000 per-thread | 4000 default, 7000 per-thread | 4000 default, 7000 per-thread |
| `cuMemcpyHtoD` | 3020 default, 7000 per-thread | 3020 default, 7000 per-thread | 3020 default, 7000 per-thread |
| `cuMemcpyDtoH` | 3020 default, 7000 per-thread | 3020 default, 7000 per-thread | 3020 default, 7000 per-thread |
| `cuMemcpyDtoD` | 3020 default, 7000 per-thread | 3020 default, 7000 per-thread | 3020 default, 7000 per-thread |
| `cuMemcpyHtoDAsync` | 3020 default, 7000 per-thread | 3020 default, 7000 per-thread | 3020 default, 7000 per-thread |
| `cuMemcpyDtoHAsync` | 3020 default, 7000 per-thread | 3020 default, 7000 per-thread | 3020 default, 7000 per-thread |
| `cuMemcpyDtoDAsync` | 3020 default, 7000 per-thread | 3020 default, 7000 per-thread | 3020 default, 7000 per-thread |
| `cuLaunchKernel` | 4000 default, 7000 per-thread | 4000 default, 7000 per-thread | 4000 default, 7000 per-thread |
| `cuLaunchKernelEx` | 11060 default, 11060 per-thread | 11060 default, 11060 per-thread | 11060 default, 11060 per-thread |
| `cuGraphLaunch` | 10000 default, 10000 per-thread | 10000 default, 10000 per-thread | 10000 default, 10000 per-thread |
| `cuStreamSynchronize` | 2000 default, 7000 per-thread | 2000 default, 7000 per-thread | 2000 default, 7000 per-thread |
| `cuCtxSynchronize` | 2000 default, 13000 default | 2000 default, 13000 default | 2000 default, 13000 default |
| `cuCtxDestroy` | — | — | — |
| `cuDevicePrimaryCtxRelease` | 7000 default | 7000 default | 7000 default |
| `cuDevicePrimaryCtxReset` | 7000 default | 7000 default | 7000 default |
| `cuFuncGetName` | 12030 default | 12030 default | 12030 default |
| `cuModuleGetFunction` | 2000 default | 2000 default | 2000 default |
| `cuLibraryGetKernel` | 12000 default | 12000 default | 12000 default |

## Requested, relevant, and not on the hook list

| Function | static | shared | pytorch |
|---|---|---|---|
| `cuDevicePrimaryCtxGetState` | 7000 default | 7000 default | 7000 default |
| `cuDevicePrimaryCtxRetain` | 7000 default | 7000 default | 7000 default |
| `cuDevicePrimaryCtxSetFlags` | 11000 default | 11000 default | 11000 default |
| `cuKernelGetFunction` | 12000 default | 12000 default | 12000 default |
| `cuLaunchCooperativeKernel` | 9000 default, 9000 per-thread | 9000 default, 9000 per-thread | 9000 default, 9000 per-thread |
| `cuLaunchHostFunc` | 10000 default, 10000 per-thread | 10000 default, 10000 per-thread | 10000 default, 10000 per-thread |
| `cuLibraryGetKernelCount` | 12040 default | 12040 default | 12040 default |
| `cuLibraryLoadData` | 12000 default | 12000 default | 12000 default |
| `cuLibraryLoadFromFile` | 12000 default | 12000 default | 12000 default |
| `cuMemAddressReserve` | — | — | 12000 default |
| `cuMemCreate` | — | — | 12000 default |
| `cuMemHostGetDevicePointer` | 3020 default | 3020 default | 3020 default |
| `cuMemHostGetFlags` | 2030 default | 2030 default | 2030 default |
| `cuMemHostRegister` | 6050 default | 6050 default | 6050 default |
| `cuMemHostUnregister` | 4000 default | 4000 default | 4000 default |
| `cuMemMap` | — | — | 12000 default |
| `cuMemPoolCreate` | 11020 default | 11020 default | 11020 default |
| `cuMemPoolDestroy` | 11020 default | 11020 default | 11020 default |
| `cuMemPoolExportPointer` | 11020 default | 11020 default | 11020 default |
| `cuMemPoolExportToShareableHandle` | 11020 default | 11020 default | 11020 default |
| `cuMemPoolGetAccess` | 11020 default | 11020 default | 11020 default |
| `cuMemPoolGetAttribute` | 11020 default | 11020 default | 11020 default |
| `cuMemPoolImportFromShareableHandle` | 11020 default | 11020 default | 11020 default |
| `cuMemPoolImportPointer` | 11020 default | 11020 default | 11020 default |
| `cuMemPoolSetAccess` | 11020 default | 11020 default | 11020 default |
| `cuMemPoolSetAttribute` | 11020 default | 11020 default | 11020 default |
| `cuMemPoolTrimTo` | 11020 default | 11020 default | 11020 default |
| `cuMemRelease` | — | — | 12000 default |
| `cuMemcpy2DAsync` | 3020 default, 7000 per-thread | 3020 default, 7000 per-thread | 3020 default, 7000 per-thread |
| `cuMemcpy2DUnaligned` | 3020 default, 7000 per-thread | 3020 default, 7000 per-thread | 3020 default, 7000 per-thread |
| `cuMemcpy3D` | 3020 default, 7000 per-thread | 3020 default, 7000 per-thread | 3020 default, 7000 per-thread |
| `cuMemcpy3DAsync` | 3020 default, 7000 per-thread | 3020 default, 7000 per-thread | 3020 default, 7000 per-thread |
| `cuMemcpy3DBatchAsync` | 12080 default, 12080 per-thread, 13000 default, 13000 per-thread | 12080 default, 12080 per-thread, 13000 default, 13000 per-thread | 12080 default, 12080 per-thread, 13000 default, 13000 per-thread |
| `cuMemcpy3DPeer` | 4000 default, 7000 per-thread | 4000 default, 7000 per-thread | 4000 default, 7000 per-thread |
| `cuMemcpy3DPeerAsync` | 4000 default, 7000 per-thread | 4000 default, 7000 per-thread | 4000 default, 7000 per-thread |
| `cuMemcpyBatchAsync` | 12080 default, 12080 per-thread, 13000 default, 13000 per-thread | 12080 default, 12080 per-thread, 13000 default, 13000 per-thread | 12080 default, 12080 per-thread, 13000 default, 13000 per-thread |
| `cuMemcpyPeer` | 4000 default, 7000 per-thread | 4000 default, 7000 per-thread | 4000 default, 7000 per-thread |
| `cuMemcpyPeerAsync` | 4000 default, 7000 per-thread | 4000 default, 7000 per-thread | 4000 default, 7000 per-thread |
| `cuMemsetD2D8` | 3020 default, 7000 per-thread | 3020 default, 7000 per-thread | 3020 default, 7000 per-thread |
| `cuMemsetD2D8Async` | 3020 default, 7000 per-thread | 3020 default, 7000 per-thread | 3020 default, 7000 per-thread |
| `cuMemsetD32Async` | — | — | 12000 default |
| `cuMemsetD8` | 3020 default, 7000 per-thread | 3020 default, 7000 per-thread | 3020 default, 7000 per-thread |
| `cuMemsetD8Async` | 3020 default, 7000 per-thread | 3020 default, 7000 per-thread | 3020 default, 7000 per-thread |
| `cuModuleLoad` | 2000 default | 2000 default | 2000 default |
| `cuModuleLoadData` | 2000 default | 2000 default | 2000 default |
| `cuModuleLoadFatBinary` | 2000 default | 2000 default | 2000 default |

## Requested with the per-thread-stream flag

67 functions: `cuEventRecord`, `cuEventRecordWithFlags`, `cuGLMapBufferObject`, `cuGLMapBufferObjectAsync`, `cuGraphInstantiateWithParams`, `cuGraphLaunch`, `cuGraphUpload`, `cuGraphicsMapResources`, `cuGraphicsUnmapResources`, `cuLaunchCooperativeKernel`, `cuLaunchHostFunc`, `cuLaunchKernel`, `cuLaunchKernelEx`, `cuMemAllocAsync`, `cuMemAllocFromPoolAsync`, `cuMemDiscardAndPrefetchBatchAsync`, `cuMemDiscardBatchAsync`, `cuMemFreeAsync`, `cuMemPrefetchAsync`, `cuMemPrefetchBatchAsync`, `cuMemcpy`, `cuMemcpy2DAsync`, `cuMemcpy2DUnaligned`, `cuMemcpy3D`, `cuMemcpy3DAsync`, `cuMemcpy3DBatchAsync`, `cuMemcpy3DPeer`, `cuMemcpy3DPeerAsync`, `cuMemcpyAsync`, `cuMemcpyBatchAsync`, `cuMemcpyDtoD`, `cuMemcpyDtoDAsync`, `cuMemcpyDtoH`, `cuMemcpyDtoHAsync`, `cuMemcpyHtoD`, `cuMemcpyHtoDAsync`, `cuMemcpyPeer`, `cuMemcpyPeerAsync`, `cuMemsetD2D8`, `cuMemsetD2D8Async`, `cuMemsetD8`, `cuMemsetD8Async`, `cuSignalExternalSemaphoresAsync`, `cuStreamAddCallback`, `cuStreamAttachMemAsync`, `cuStreamBatchMemOp`, `cuStreamBeginCapture`, `cuStreamBeginCaptureToGraph`, `cuStreamCopyAttributes`, `cuStreamEndCapture`, `cuStreamGetAttribute`, `cuStreamGetCaptureInfo`, `cuStreamGetDevice`, `cuStreamGetFlags`, `cuStreamGetId`, `cuStreamGetPriority`, `cuStreamIsCapturing`, `cuStreamQuery`, `cuStreamSetAttribute`, `cuStreamSynchronize`, `cuStreamUpdateCaptureDependencies`, `cuStreamWaitEvent`, `cuStreamWaitValue32`, `cuStreamWaitValue64`, `cuStreamWriteValue32`, `cuStreamWriteValue64`, `cuWaitExternalSemaphoresAsync`

