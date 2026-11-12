// The implementations llvm-libc's memory benchmark measures: this file
// replaces its libc/benchmarks/LibcDefaultImplementations.cpp, the harness's
// list of implementations to test (prepare.py copies it in before the build).
//
// Each function is measured as:
//   llvm-libc  LIBC_NAMESPACE::<fn>, built for this host (the harness's own)
//   glibc      the system C library's <fn>, through its IFUNC selection
//   rep        `rep movsb` / `rep stosb` (memcpy, memset, bzero only)
//   base       base's own CLIF (lean/algorithms/Bench/Memops.lean), compiled
//              by base's driver and called through its C API (base.h): the
//              library BASE_LIB names, loading the artifact BASE_MEMOPS names
//              (memcpy, memset, bzero, memcmp, bcmp; memmove is not one, as
//              base's functions take an input and an output that do not
//              overlap). Each call goes through a wrapper here that puts the
//              arguments in the C API's order and jumps to base's function
//              (base's C API takes (in, in_len, out, out_len), not a libc
//              signature).
// Rust's core and std call the C library's functions on Linux, so glibc is
// what a Rust program gets.
//
// Before a list is first handed to the harness, every implementation in it
// is checked against a byte loop over every size 0..1024 at several offsets;
// a mismatch aborts the run before anything is timed.

#include "LibcFunctionPrototypes.h"
#include "src/__support/macros/config.h"
#include "llvm/ADT/ArrayRef.h"
#include <cstddef>
#include <cstdint>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <strings.h>

namespace LIBC_NAMESPACE_DECL {

extern void *memcpy(void *__restrict, const void *__restrict, size_t);
extern void *memmove(void *, const void *, size_t);
extern void *memset(void *, int, size_t);
extern void bzero(void *, size_t);
extern int memcmp(const void *, const void *, size_t);
extern int bcmp(const void *, const void *, size_t);

} // namespace LIBC_NAMESPACE_DECL

using llvm::libc_benchmarks::BzeroConfiguration;
using llvm::libc_benchmarks::MemcmpOrBcmpConfiguration;
using llvm::libc_benchmarks::MemcpyConfiguration;
using llvm::libc_benchmarks::MemmoveConfiguration;
using llvm::libc_benchmarks::MemsetConfiguration;

static void *rep_movsb(void *__restrict dst, const void *__restrict src, size_t n) {
  void *ret = dst;
  asm volatile("rep movsb" : "+D"(dst), "+S"(src), "+c"(n) : : "memory");
  return ret;
}

static void *rep_stosb(void *dst, int c, size_t n) {
  void *ret = dst;
  asm volatile("rep stosb" : "+D"(dst), "+c"(n) : "a"(c) : "memory");
  return ret;
}

static void rep_stosb_zero(void *dst, size_t n) { rep_stosb(dst, 0, n); }

static void *glibc_memcpy(void *__restrict d, const void *__restrict s, size_t n) { return ::memcpy(d, s, n); }
static void *glibc_memmove(void *d, const void *s, size_t n) { return ::memmove(d, s, n); }
static void *glibc_memset(void *d, int c, size_t n) { return ::memset(d, c, n); }
static void glibc_bzero(void *d, size_t n) { ::bzero(d, n); }
static int glibc_memcmp(const void *a, const void *b, size_t n) { return ::memcmp(a, b, n); }
static int glibc_bcmp(const void *a, const void *b, size_t n) { return ::bcmp(a, b, n); }

// ---- base, through its C API ----------------------------------------------

typedef int64_t (*BaseFunction)(const uint8_t *, size_t, uint8_t *, size_t);

struct BaseOps {
  BaseFunction memcpy_, memset_, bzero_, memcmp_, bcmp_;
};

[[noreturn]] static void base_fail(const char *what) {
  std::fprintf(stderr, "base: %s\n", what);
  std::abort();
}

// The functions, from the driver BASE_LIB's library makes of the artifact at
// BASE_MEMOPS, loaded by the first list asked for (every list is asked for
// before anything is timed), so a call reads a plain global and checks no
// guard.
static BaseOps g_base;

static void base_load() {
  if (g_base.memcpy_) return;
  g_base = [] {
    const char *lib = std::getenv("BASE_LIB"), *art = std::getenv("BASE_MEMOPS");
    if (!lib || !art) base_fail("BASE_LIB and BASE_MEMOPS name base's library and its memops artifact");
    void *h = dlopen(lib, RTLD_NOW);
    if (!h) base_fail(dlerror());
    using Load = void *(*)(const uint8_t *, size_t, char *, size_t);
    using Function = BaseFunction (*)(const void *, const uint8_t *, size_t);
    auto load = (Load)dlsym(h, "base_driver_load");
    auto function = (Function)dlsym(h, "base_driver_function");
    if (!load || !function) base_fail("the library has no base_driver_load / base_driver_function");
    FILE *f = std::fopen(art, "rb");
    if (!f) base_fail("cannot open BASE_MEMOPS");
    std::fseek(f, 0, SEEK_END);
    long len = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    auto *bytes = (uint8_t *)std::malloc(len);
    if (std::fread(bytes, 1, len, f) != (size_t)len) base_fail("cannot read BASE_MEMOPS");
    std::fclose(f);
    char err[512];
    void *drv = load(bytes, len, err, sizeof err);
    if (!drv) base_fail(err);
    auto fn = [&](const char *name) {
      BaseFunction p = function(drv, (const uint8_t *)name, std::strlen(name));
      if (!p) base_fail(name);
      return p;
    };
    return BaseOps{fn("memcpy"), fn("memset"), fn("bzero"), fn("memcmp"), fn("bcmp")};
  }();
}

// base's memcpy and memset answer the output's address, as libc's answer
// their destination, so each wrapper is one tail call; memset's byte is read
// from a table of every byte value.
static void *base_memcpy(void *__restrict d, const void *__restrict s, size_t n) {
  return (void *)g_base.memcpy_((const uint8_t *)s, n, (uint8_t *)d, n);
}
static constexpr auto kBytes = [] {
  struct { uint8_t b[256]; } t{};
  for (int i = 0; i < 256; ++i) t.b[i] = (uint8_t)i;
  return t;
}();
static void *base_memset(void *d, int c, size_t n) {
  return (void *)g_base.memset_(&kBytes.b[(uint8_t)c], 1, (uint8_t *)d, n);
}
static void base_bzero(void *d, size_t n) { g_base.bzero_(nullptr, 0, (uint8_t *)d, n); }
// The second buffer is the C API's output, which these only read.
static int base_memcmp(const void *a, const void *b, size_t n) {
  return (int)g_base.memcmp_((const uint8_t *)a, n, (uint8_t *)const_cast<void *>(b), n);
}
static int base_bcmp(const void *a, const void *b, size_t n) {
  return (int)g_base.bcmp_((const uint8_t *)a, n, (uint8_t *)const_cast<void *>(b), n);
}

// ---- the checks ----------------------------------------------------------

static constexpr size_t kMax = 1024, kPad = 64, kBuf = kMax + 2 * kPad;

[[noreturn]] static void fail(const char *fn, const char *name, size_t n, size_t off) {
  std::fprintf(stderr, "CHECK FAILED: %s (%s) at size %zu offset %zu\n", fn, name, n, off);
  std::abort();
}

static void fill(unsigned char *p, size_t n, unsigned seed) {
  for (size_t i = 0; i < n; ++i) p[i] = (unsigned char)(i * 131 + seed * 17 + 7);
}

static void check(const MemcpyConfiguration &c) {
  unsigned char src[kBuf], dst[kBuf], ref[kBuf];
  for (size_t off = 0; off < 8; off += 3)
    for (size_t n = 0; n <= kMax; ++n) {
      fill(src, kBuf, 1); fill(dst, kBuf, 2); fill(ref, kBuf, 2);
      for (size_t i = 0; i < n; ++i) ref[kPad + off + i] = src[kPad + i];
      if (c.function(dst + kPad + off, src + kPad, n) != dst + kPad + off || std::memcmp(dst, ref, kBuf))
        fail("memcpy", c.name.data(), n, off);
    }
}

static void check(const MemmoveConfiguration &c) {
  unsigned char buf[kBuf * 2], ref[kBuf * 2], tmp[kBuf];
  for (long shift : {-37L, -1L, 1L, 37L})
    for (size_t n = 0; n <= kMax; ++n) {
      fill(buf, sizeof buf, 3); fill(ref, sizeof ref, 3);
      unsigned char *s = buf + kBuf / 2, *d = s + shift;
      for (size_t i = 0; i < n; ++i) tmp[i] = ref[kBuf / 2 + i];
      for (size_t i = 0; i < n; ++i) ref[kBuf / 2 + shift + i] = tmp[i];
      if (c.function(d, s, n) != d || std::memcmp(buf, ref, sizeof buf))
        fail("memmove", c.name.data(), n, (size_t)(shift + 64));
    }
}

static void check(const MemsetConfiguration &c) {
  unsigned char dst[kBuf], ref[kBuf];
  for (size_t off = 0; off < 8; off += 3)
    for (size_t n = 0; n <= kMax; ++n) {
      fill(dst, kBuf, 4); fill(ref, kBuf, 4);
      for (size_t i = 0; i < n; ++i) ref[kPad + off + i] = (unsigned char)0xA5;
      if (c.function(dst + kPad + off, 0xA5, n) != dst + kPad + off || std::memcmp(dst, ref, kBuf))
        fail("memset", c.name.data(), n, off);
    }
}

static void check(const BzeroConfiguration &c) {
  unsigned char dst[kBuf], ref[kBuf];
  for (size_t off = 0; off < 8; off += 3)
    for (size_t n = 0; n <= kMax; ++n) {
      fill(dst, kBuf, 5); fill(ref, kBuf, 5);
      for (size_t i = 0; i < n; ++i) ref[kPad + off + i] = 0;
      c.function(dst + kPad + off, n);
      if (std::memcmp(dst, ref, kBuf)) fail("bzero", c.name.data(), n, off);
    }
}

static int sign(int x) { return (x > 0) - (x < 0); }

static void check(const MemcmpOrBcmpConfiguration &c, bool ordered) {
  unsigned char a[kBuf], b[kBuf];
  for (size_t n = 0; n <= kMax; ++n)
    for (size_t at = 0; at <= n; at += (n < 64 ? 1 : 61)) {
      fill(a, kBuf, 6); fill(b, kBuf, 6);
      int want = 0;
      if (at < n) { b[kPad + at] = (unsigned char)(a[kPad + at] + 1); want = a[kPad + at] < b[kPad + at] ? -1 : 1; }
      int got = c.function(a + kPad, b + kPad, n);
      if (ordered ? sign(got) != want : (got != 0) != (want != 0)) fail(ordered ? "memcmp" : "bcmp", c.name.data(), n, at);
    }
}

template <typename T, size_t N> static llvm::ArrayRef<T> checked(const T (&list)[N]) {
  for (const T &c : list) check(c);
  return llvm::ArrayRef(list);
}

// ---- the lists -------------------------------------------------------------

llvm::ArrayRef<MemcpyConfiguration> getMemcpyConfigurations() {
  base_load();
  static constexpr MemcpyConfiguration list[] = {
      {LIBC_NAMESPACE::memcpy, "llvm-libc"}, {glibc_memcpy, "glibc"}, {rep_movsb, "rep"}, {base_memcpy, "base"}};
  static const auto ref = checked(list);
  return ref;
}
llvm::ArrayRef<MemmoveConfiguration> getMemmoveConfigurations() {
  static constexpr MemmoveConfiguration list[] = {{LIBC_NAMESPACE::memmove, "llvm-libc"}, {glibc_memmove, "glibc"}};
  static const auto ref = checked(list);
  return ref;
}
llvm::ArrayRef<MemcmpOrBcmpConfiguration> getMemcmpConfigurations() {
  base_load();
  static constexpr MemcmpOrBcmpConfiguration list[] = {{LIBC_NAMESPACE::memcmp, "llvm-libc"}, {glibc_memcmp, "glibc"}, {base_memcmp, "base"}};
  static const auto ref = [] { for (const auto &c : list) check(c, true); return llvm::ArrayRef(list); }();
  return ref;
}
llvm::ArrayRef<MemcmpOrBcmpConfiguration> getBcmpConfigurations() {
  base_load();
  static constexpr MemcmpOrBcmpConfiguration list[] = {{LIBC_NAMESPACE::bcmp, "llvm-libc"}, {glibc_bcmp, "glibc"}, {base_bcmp, "base"}};
  static const auto ref = [] { for (const auto &c : list) check(c, false); return llvm::ArrayRef(list); }();
  return ref;
}
llvm::ArrayRef<MemsetConfiguration> getMemsetConfigurations() {
  base_load();
  static constexpr MemsetConfiguration list[] = {
      {LIBC_NAMESPACE::memset, "llvm-libc"}, {glibc_memset, "glibc"}, {rep_stosb, "rep"}, {base_memset, "base"}};
  static const auto ref = checked(list);
  return ref;
}
llvm::ArrayRef<BzeroConfiguration> getBzeroConfigurations() {
  base_load();
  static constexpr BzeroConfiguration list[] = {
      {LIBC_NAMESPACE::bzero, "llvm-libc"}, {glibc_bzero, "glibc"}, {rep_stosb_zero, "rep"}, {base_bzero, "base"}};
  static const auto ref = checked(list);
  return ref;
}
