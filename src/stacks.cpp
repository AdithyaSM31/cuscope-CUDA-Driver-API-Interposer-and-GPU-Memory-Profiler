#include "stacks.hpp"

#include <cxxabi.h>
#include <dlfcn.h>
#include <elf.h>
#include <execinfo.h>
#include <fcntl.h>
#include <link.h>
#include <sys/auxv.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "config.hpp"

namespace cuscope {
namespace {

constexpr int kMaxDepth = 64;
constexpr int kMaxSkip = 16;  // frames dropped before the stack starts

// ---- Capture ----

std::mutex g_stacks_mu;

std::unordered_map<uint32_t, std::vector<uintptr_t>>& stacks() {
  static auto* s = new std::unordered_map<uint32_t, std::vector<uintptr_t>>();
  return *s;
}

int resolve_depth(int depth) {
  if (depth < 0) depth = config().stack_depth;
  return std::clamp(depth, 1, kMaxDepth);
}

// Hashes the frames, then finds or adds the stack under that id. On a hash
// collision with a different stack, probes the next id.
uint32_t store(void* const* frames, int n) {
  if (n <= 0) return 0;
  uint64_t h = 0xcbf29ce484222325ull;  // FNV-1a over the frame addresses
  for (int i = 0; i < n; ++i) {
    h ^= reinterpret_cast<uintptr_t>(frames[i]);
    h *= 0x100000001b3ull;
  }
  uint32_t id = static_cast<uint32_t>(h ^ (h >> 32));

  std::lock_guard<std::mutex> lock(g_stacks_mu);
  auto& all = stacks();
  for (;; ++id) {
    if (id == 0) continue;  // 0 means "no stack"
    auto it = all.find(id);
    if (it == all.end()) {
      std::vector<uintptr_t>& v = all[id];
      for (int i = 0; i < n; ++i) v.push_back(reinterpret_cast<uintptr_t>(frames[i]));
      return id;
    }
    const std::vector<uintptr_t>& v = it->second;
    if (static_cast<int>(v.size()) == n &&
        std::equal(v.begin(), v.end(), frames, [](uintptr_t a, void* b) { return a == reinterpret_cast<uintptr_t>(b); })) {
      return id;
    }
  }
}

// Address range of the loaded object containing addr.
struct ModuleRange {
  uintptr_t lo = 0, hi = 0;
  bool contains(const void* p) const {
    auto a = reinterpret_cast<uintptr_t>(p);
    return a >= lo && a < hi;
  }
};

ModuleRange range_of(const void* addr) {
  struct Search {
    uintptr_t addr;
    ModuleRange found;
  } search{reinterpret_cast<uintptr_t>(addr), {}};
  dl_iterate_phdr(
      [](dl_phdr_info* info, size_t, void* data) -> int {
        auto* s = static_cast<Search*>(data);
        ModuleRange r{UINTPTR_MAX, 0};
        for (int i = 0; i < info->dlpi_phnum; ++i) {
          const ElfW(Phdr)& ph = info->dlpi_phdr[i];
          if (ph.p_type != PT_LOAD) continue;
          r.lo = std::min<uintptr_t>(r.lo, info->dlpi_addr + ph.p_vaddr);
          r.hi = std::max<uintptr_t>(r.hi, info->dlpi_addr + ph.p_vaddr + ph.p_memsz);
        }
        if (s->addr < r.lo || s->addr >= r.hi) return 0;
        s->found = r;
        return 1;
      },
      &search);
  return search.found;
}

const ModuleRange& cuscope_range() {
  static const ModuleRange r = range_of(reinterpret_cast<const void*>(&capture_caller_stack));
  return r;
}

// ---- Symbolisation ----

// FUNC symbols from an ELF file's .symtab, read through a read-only mapping.
class ElfSymbols {
 public:
  explicit ElfSymbols(const std::string& path) {
    int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) return;
    struct stat st {};
    void* map = (::fstat(fd, &st) == 0 && st.st_size > 0)
                    ? ::mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0)
                    : MAP_FAILED;
    ::close(fd);
    if (map == MAP_FAILED) return;
    parse(static_cast<const unsigned char*>(map), static_cast<size_t>(st.st_size));
    ::munmap(map, st.st_size);
  }

  // Name of the function containing `offset` (relative to the module's load
  // address, which dladdr reports as the start of its first segment), or null.
  const char* find(uintptr_t offset) const {
    const uintptr_t key = offset + first_load_vaddr_;  // the address the ELF file uses
    auto it = std::upper_bound(syms_.begin(), syms_.end(), key,
                               [](uintptr_t k, const Sym& s) { return k < s.value; });
    if (it == syms_.begin()) return nullptr;
    --it;
    return key < it->value + std::max<uint64_t>(it->size, 1) ? it->name.c_str() : nullptr;
  }

 private:
  struct Sym {
    uint64_t value, size;
    std::string name;
  };

  void parse(const unsigned char* data, size_t size) {
    if (size < sizeof(ElfW(Ehdr)) || std::memcmp(data, ELFMAG, SELFMAG) != 0 || data[EI_CLASS] != ELFCLASS64) {
      return;
    }
    const auto* eh = reinterpret_cast<const ElfW(Ehdr)*>(data);

    // dladdr's base is the page-aligned start of the first PT_LOAD segment.
    if (eh->e_phoff + static_cast<uint64_t>(eh->e_phnum) * sizeof(ElfW(Phdr)) <= size) {
      const auto* ph = reinterpret_cast<const ElfW(Phdr)*>(data + eh->e_phoff);
      for (int i = 0; i < eh->e_phnum; ++i) {
        if (ph[i].p_type == PT_LOAD) {
          const auto page = static_cast<uintptr_t>(sysconf(_SC_PAGESIZE));
          first_load_vaddr_ = ph[i].p_vaddr & ~(page - 1);
          break;
        }
      }
    }

    if (eh->e_shoff == 0 || eh->e_shoff + static_cast<uint64_t>(eh->e_shnum) * sizeof(ElfW(Shdr)) > size) return;
    const auto* sh = reinterpret_cast<const ElfW(Shdr)*>(data + eh->e_shoff);
    for (int i = 0; i < eh->e_shnum; ++i) {
      if (sh[i].sh_type != SHT_SYMTAB || sh[i].sh_link >= eh->e_shnum) continue;
      const ElfW(Shdr)& strtab = sh[sh[i].sh_link];
      if (sh[i].sh_offset + sh[i].sh_size > size || strtab.sh_offset + strtab.sh_size > size) return;
      const auto* syms = reinterpret_cast<const ElfW(Sym)*>(data + sh[i].sh_offset);
      const char* names = reinterpret_cast<const char*>(data + strtab.sh_offset);
      for (size_t j = 0; j < sh[i].sh_size / sizeof(ElfW(Sym)); ++j) {
        const ElfW(Sym)& s = syms[j];
        if (ELF64_ST_TYPE(s.st_info) != STT_FUNC || s.st_value == 0 || s.st_name >= strtab.sh_size) continue;
        syms_.push_back({s.st_value, s.st_size, names + s.st_name});
      }
    }
    std::sort(syms_.begin(), syms_.end(), [](const Sym& a, const Sym& b) { return a.value < b.value; });
  }

  std::vector<Sym> syms_;
  uintptr_t first_load_vaddr_ = 0;
};

// dladdr names the main program with argv[0], which may be relative; use the real path.
std::string module_path(const Dl_info& info) {
  static const void* main_base = [] {
    Dl_info main_info{};
    return dladdr(reinterpret_cast<const void*>(getauxval(AT_PHDR)), &main_info) ? main_info.dli_fbase : nullptr;
  }();
  if (info.dli_fbase == main_base || info.dli_fname == nullptr || info.dli_fname[0] != '/') {
    char path[PATH_MAX];
    ssize_t n = ::readlink("/proc/self/exe", path, sizeof(path) - 1);
    if (n > 0) return std::string(path, n);
  }
  return info.dli_fname ? info.dli_fname : "";
}

StackFrame symbolize(uintptr_t addr, std::unordered_map<std::string, std::unique_ptr<ElfSymbols>>& elf) {
  StackFrame frame{"", addr, ""};
  Dl_info info{};
  if (dladdr(reinterpret_cast<void*>(addr), &info) == 0) return frame;
  frame.module = module_path(info);
  frame.offset = addr - reinterpret_cast<uintptr_t>(info.dli_fbase);

  auto& symbols = elf[frame.module];
  if (!symbols) symbols = std::make_unique<ElfSymbols>(frame.module);
  // A return address points just past the call; look up the call itself.
  const char* name = symbols->find(frame.offset - 1);
  if (name == nullptr) name = info.dli_sname;
  if (name != nullptr) frame.symbol = demangle(name);
  return frame;
}

// Index in raw of the frame that called the capture function: the entry equal
// to that function's own return address. Anchoring on it, rather than assuming
// "one frame up", stays right when something sits above it on the stack (a
// sanitizer intercepts backtrace() itself). -1 if it is not there.
int caller_frame(void* const* raw, int n, const void* return_address) {
  for (int i = 0; i < n; ++i) {
    if (raw[i] == return_address) return i;
  }
  return -1;
}

}  // namespace

__attribute__((noinline)) uint32_t capture_stack(int skip, int depth) {
  depth = resolve_depth(depth);
  skip = std::clamp(skip, 0, kMaxSkip);
  void* raw[kMaxDepth + 2 * kMaxSkip];
  const int n = backtrace(raw, depth + skip + kMaxSkip);
  int first = caller_frame(raw, n, __builtin_return_address(0));
  if (first < 0) first = 1;  // raw[0] is inside capture_stack
  first = std::min(n, first + skip);
  return store(raw + first, std::min(n - first, depth));
}

__attribute__((noinline)) uint32_t capture_caller_stack() {
  const int depth = resolve_depth(-1);
  void* raw[kMaxDepth + kMaxSkip];
  const int n = backtrace(raw, depth + kMaxSkip);
  const ModuleRange& self = cuscope_range();
  int first = std::max(0, caller_frame(raw, n, __builtin_return_address(0)));
  while (first < n && self.contains(raw[first])) ++first;
  return store(raw + first, std::min(n - first, depth));
}

std::map<uint32_t, std::vector<StackFrame>> symbolize_all() {
  std::unordered_map<uint32_t, std::vector<uintptr_t>> copy;
  {
    std::lock_guard<std::mutex> lock(g_stacks_mu);
    copy = stacks();
  }
  std::unordered_map<std::string, std::unique_ptr<ElfSymbols>> elf;
  std::map<uint32_t, std::vector<StackFrame>> out;
  for (const auto& [id, frames] : copy) {
    std::vector<StackFrame>& dst = out[id];
    for (uintptr_t addr : frames) dst.push_back(symbolize(addr, elf));
  }
  return out;
}

std::string demangle(const char* mangled) {
  if (mangled == nullptr) return "";
  int status = 0;
  char* out = abi::__cxa_demangle(mangled, nullptr, nullptr, &status);
  if (status != 0 || out == nullptr) return mangled;
  std::string result(out);
  std::free(out);
  return result;
}

void warm_up_backtrace() {
  void* frames[2];
  backtrace(frames, 2);
}

void stacks_before_fork() { g_stacks_mu.lock(); }
void stacks_after_fork() { g_stacks_mu.unlock(); }

}  // namespace cuscope
