// Protobuf parsing and serializing: base's generated codec
// (lean/algorithms/Bench/Proto.lean, from the benchmark's own .proto) beside
// protobuf C++ (with an arena and without) and upb, on HyperProtoBench's
// messages: each top-level message of a benchmark built by the benchmark's own
// generated Set function, as its driver builds them.
//
//   bench <libbase.so> <proto.cbor> <name>        # a JSON line each
//
// Built once per benchmark, against its generated protobuf and upb code; the
// top-level messages come in as the TOPS X-macro (tops.h, from the schema).
// Before anything is timed:
//  - protobuf's serialization of every message is the corpus;
//  - upb parses every message and serializes it to the same bytes, and
//    protobuf without an arena parses and re-serializes it to them too;
//  - base decodes the whole corpus and encodes it to exactly the corpus, and
//    protobuf parses base's encoding to the same message (its bytes again).
// Then each engine parses the corpus and serializes it REPS times after one
// untimed pass, every time printed.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <string>
#include <vector>

#include "benchmark.pb.h"
#include "benchmark.upb.h"
#include "tops.h"

// HyperProtoBench's driver (benchmark.cc, Apache 2.0): what its generated Get
// functions hand their values to, so the generated code compiles as it does
// there.
namespace hyperprotobench {
const size_t kReceiverSize = 65536;
struct Receiver {
  std::array<const std::string *, kReceiverSize> string_r;
  size_t string_i;
  std::array<bool, kReceiverSize> bool_r;
  size_t bool_i;
  std::array<double, kReceiverSize> double_r;
  size_t double_i;
  std::array<float, kReceiverSize> float_r;
  size_t float_i;
  std::array<int32_t, kReceiverSize> int32_r;
  size_t int32_i;
  std::array<int64_t, kReceiverSize> int64_r;
  size_t int64_i;
  std::array<uint32_t, kReceiverSize> uint32_r;
  size_t uint32_i;
  std::array<uint64_t, kReceiverSize> uint64_r;
  size_t uint64_i;
};
struct Receiver receiver;
#define RECEIVE(name, type, field)                                                                                     \
  inline void name(const type &val) {                                                                                  \
    receiver.field##_i = (receiver.field##_i + 1) % kReceiverSize;                                                     \
    receiver.field##_r[receiver.field##_i] = val;                                                                      \
  }
inline void ReceiveString(const std::string &val) {
  receiver.string_i = (receiver.string_i + 1) % kReceiverSize;
  receiver.string_r[receiver.string_i] = &val;
}
RECEIVE(ReceiveBool, bool, bool)
RECEIVE(ReceiveDouble, double, double)
RECEIVE(ReceiveFloat, float, float)
RECEIVE(ReceiveInt32, int32_t, int32)
RECEIVE(ReceiveInt64, int64_t, int64)
RECEIVE(ReceiveUint32, uint32_t, uint32)
RECEIVE(ReceiveUint64, uint64_t, uint64)
#include "benchmark.inc"
} // namespace hyperprotobench

using namespace hyperprotobench;
using google::protobuf::Arena;

static constexpr int REPS = 200;
typedef int64_t (*BaseFunction)(const uint8_t *, size_t, uint8_t *, size_t);

[[noreturn]] static void die(const char *what) {
  std::fprintf(stderr, "%s\n", what);
  std::exit(1);
}

static double now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static std::string readFile(const char *path) {
  FILE *f = std::fopen(path, "rb");
  if (!f) die(path);
  std::string s;
  char buf[1 << 16];
  size_t n;
  while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) s.append(buf, n);
  std::fclose(f);
  return s;
}

// The messages, as protobuf serializes them, and framed for base:
// (type, length, bytes), the type its place in TOPS.
struct Corpus {
  std::vector<std::string> msgs;
  std::string framed;
  size_t bytes = 0;
};

static void emit(const char *bench, const char *engine, const char *op, size_t bytes, const std::vector<double> &ts) {
  std::vector<double> s = ts;
  std::sort(s.begin(), s.end());
  std::printf("{\"bench\": \"%s\", \"engine\": \"%s\", \"op\": \"%s\", \"bytes\": %zu, \"seconds\": %.9f, \"samples\": [",
              bench, engine, op, bytes, s[s.size() / 2]);
  for (size_t i = 0; i < ts.size(); ++i) std::printf("%s%.9f", i ? ", " : "", ts[i]);
  std::printf("]}\n");
}

template <typename F> static std::vector<double> times(F f) {
  f();
  std::vector<double> ts;
  for (int r = 0; r < REPS; ++r) {
    double t = now();
    f();
    ts.push_back(now() - t);
  }
  return ts;
}

int main(int argc, char **argv) {
  if (argc < 4) die("usage: bench <libbase.so> <proto.cbor> <name>");
  const char *name = argv[3];
  void *lib = dlopen(argv[1], RTLD_NOW);
  if (!lib) die(dlerror());
  auto load = (void *(*)(const uint8_t *, size_t, char *, size_t))dlsym(lib, "base_driver_load");
  auto function = (BaseFunction(*)(const void *, const uint8_t *, size_t))dlsym(lib, "base_driver_function");
  std::string art = readFile(argv[2]);
  char err[512];
  void *drv = load((const uint8_t *)art.data(), art.size(), err, sizeof err);
  if (!drv) die(err);
  BaseFunction decode = function(drv, (const uint8_t *)"decode", 6), encode = function(drv, (const uint8_t *)"encode", 6);
  if (!decode || !encode) die("the artifact has no decode or encode");

  // the corpus: each top-level message as the benchmark's Set builds it
  Corpus c;
  Arena build;
  uint32_t ty = 0;
#define TOP(T)                                                                                                         \
  {                                                                                                                    \
    T *m = Arena::Create<T>(&build), *other = Arena::Create<T>(&build);                                                \
    std::string s;                                                                                                     \
    T##_Set_F1(m, other, &s);                                                                                          \
    std::string bytes = m->SerializeAsString();                                                                        \
    uint32_t hdr[2] = {ty++, (uint32_t)bytes.size()};                                                                  \
    c.framed.append((const char *)hdr, 8);                                                                             \
    c.framed += bytes;                                                                                                 \
    c.bytes += bytes.size();                                                                                           \
    c.msgs.push_back(std::move(bytes));                                                                                \
  }
  TOPS
#undef TOP

  // the checks
  {
    upb_Arena *ua = upb_Arena_New();
    size_t i = 0;
#define TOP(T)                                                                                                         \
  {                                                                                                                    \
    const std::string &b = c.msgs[i++];                                                                                \
    auto *u = hyperprotobench_##T##_parse(b.data(), b.size(), ua);                                                     \
    size_t n;                                                                                                          \
    char *o = u ? hyperprotobench_##T##_serialize(u, ua, &n) : nullptr;                                                \
    if (!o || std::string(o, n) != b) die("upb does not round-trip " #T);                                              \
    T p;                                                                                                               \
    if (!p.ParseFromString(b) || p.SerializeAsString() != b) die("protobuf does not round-trip " #T);                 \
  }
    TOPS
#undef TOP
    upb_Arena_Free(ua);
    if (decode((const uint8_t *)c.framed.data(), c.framed.size(), nullptr, 0) != 0) die("base refused the corpus");
    std::string out(c.framed.size() + 64, '\0');
    if (encode(nullptr, 0, (uint8_t *)out.data(), out.size()) != 0) die("base's encode failed");
    if (std::memcmp(out.data(), c.framed.data(), c.framed.size()) != 0) die("base's encoding is not the corpus");
    // protobuf reads base's bytes back as the same message
    size_t at = 0;
    i = 0;
#define TOP(T)                                                                                                         \
  {                                                                                                                    \
    uint32_t n;                                                                                                        \
    std::memcpy(&n, out.data() + at + 4, 4);                                                                           \
    T p;                                                                                                               \
    if (!p.ParseFromArray(out.data() + at + 8, (int)n) || p.SerializeAsString() != c.msgs[i++])                       \
      die("protobuf reads base's " #T " as another message");                                                          \
    at += 8 + n;                                                                                                       \
  }
    TOPS
#undef TOP
  }

  // protobuf, its messages on an arena
  emit(name, "protobuf-arena", "parse", c.bytes, times([&] {
         Arena a;
         size_t i = 0;
#define TOP(T) Arena::Create<T>(&a)->ParseFromString(c.msgs[i++]);
         TOPS
#undef TOP
       }));
  // protobuf, each message its own
  emit(name, "protobuf", "parse", c.bytes, times([&] {
         size_t i = 0;
#define TOP(T)                                                                                                         \
  {                                                                                                                    \
    T m;                                                                                                               \
    m.ParseFromString(c.msgs[i++]);                                                                                    \
  }
         TOPS
#undef TOP
       }));
  {
    Arena a;
    std::vector<google::protobuf::Message *> parsed;
    size_t i = 0;
#define TOP(T)                                                                                                         \
  {                                                                                                                    \
    T *m = Arena::Create<T>(&a);                                                                                       \
    m->ParseFromString(c.msgs[i++]);                                                                                   \
    parsed.push_back(m);                                                                                               \
  }
    TOPS
#undef TOP
    std::string buf(c.bytes + 64, '\0');
    emit(name, "protobuf-arena", "serialize", c.bytes, times([&] {
           size_t at = 0;
           for (auto *m : parsed) {
             size_t n = m->ByteSizeLong();
             m->SerializeToArray(buf.data() + at, (int)n);
             at += n;
           }
         }));
  }
  // upb
  emit(name, "upb", "parse", c.bytes, times([&] {
         upb_Arena *ua = upb_Arena_New();
         size_t i = 0;
#define TOP(T) hyperprotobench_##T##_parse(c.msgs[i].data(), c.msgs[i].size(), ua), ++i;
         TOPS
#undef TOP
         upb_Arena_Free(ua);
       }));
  {
    upb_Arena *keep = upb_Arena_New();
    std::vector<const void *> parsed;
    size_t i = 0;
#define TOP(T) parsed.push_back(hyperprotobench_##T##_parse(c.msgs[i].data(), c.msgs[i].size(), keep)), ++i;
    TOPS
#undef TOP
    emit(name, "upb", "serialize", c.bytes, times([&] {
           upb_Arena *ua = upb_Arena_New();
           size_t j = 0, n;
#define TOP(T) hyperprotobench_##T##_serialize((const hyperprotobench_##T *)parsed[j++], ua, &n);
           TOPS
#undef TOP
           upb_Arena_Free(ua);
         }));
    upb_Arena_Free(keep);
  }
  // base: the whole corpus a call, as it decodes and encodes it
  emit(name, "base", "parse", c.bytes, times([&] {
         if (decode((const uint8_t *)c.framed.data(), c.framed.size(), nullptr, 0) != 0) die("base refused the corpus");
       }));
  {
    std::string out(c.framed.size() + 64, '\0');
    emit(name, "base", "serialize", c.bytes, times([&] {
           if (encode(nullptr, 0, (uint8_t *)out.data(), out.size()) != 0) die("base's encode failed");
         }));
  }
  return 0;
}
