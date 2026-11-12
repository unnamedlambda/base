// JSON parsing: base's parser (lean/algorithms/Bench/Json.lean) beside
// simdjson's DOM parser, on the standard corpora, measured as simdjson
// measures itself: bytes of text parsed a second, the parse alone (simdjson's
// input padded once beforehand, as its own benchmarks do).
//
//   bench <libbase.so> <json.cbor> <file.json>...      # a JSON line each
//
// Before any timing, each file's checksum over every value in document order
// is computed both ways, by base's `sum` over its tape and by the same fold
// over simdjson's DOM, and must agree. Each engine then parses each file REPS
// times after one untimed parse; every time is printed.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <string>
#include <time.h>
#include <vector>

#include "simdjson.h"

typedef int64_t (*BaseFunction)(const uint8_t *, size_t, uint8_t *, size_t);

static constexpr int REPS = 50;

static double now() {
  timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
}

[[noreturn]] static void die(const char *what) {
  std::fprintf(stderr, "%s\n", what);
  std::exit(1);
}

// The checksum base's `sum` computes, over simdjson's DOM.
struct Sum {
  uint64_t h = 0xCBF29CE484222325ull;
  void mix(uint64_t x) { h = (h ^ x) * 0x100000001B3ull; }
  void str(std::string_view s) {
    mix('"');
    mix(s.size());
    for (unsigned char b : s) mix(b);
  }
  void value(simdjson::dom::element e) {
    using simdjson::dom::element_type;
    switch (e.type()) {
    case element_type::OBJECT:
      mix('{');
      for (auto f : simdjson::dom::object(e)) {
        str(f.key);
        value(f.value);
      }
      mix('}');
      break;
    case element_type::ARRAY:
      mix('[');
      for (auto v : simdjson::dom::array(e)) value(v);
      mix(']');
      break;
    case element_type::STRING: str(std::string_view(e)); break;
    case element_type::INT64: mix('l'); mix(uint64_t(int64_t(e))); break;
    case element_type::UINT64: mix('u'); mix(uint64_t(e)); break;
    case element_type::DOUBLE: {
      double d = double(e);
      uint64_t bits;
      std::memcpy(&bits, &d, 8);
      mix('d');
      mix(bits);
      break;
    }
    case element_type::BOOL: mix(bool(e) ? 't' : 'f'); break;
    case element_type::NULL_VALUE: mix('n'); break;
    case element_type::BIGINT: die("a big integer, which base refuses");
    }
  }
};

static std::string read(const char *path) {
  FILE *f = std::fopen(path, "rb");
  if (!f) die(path);
  std::string s;
  char buf[1 << 16];
  size_t n;
  while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) s.append(buf, n);
  std::fclose(f);
  return s;
}

static void emit(const char *engine, const char *file, size_t bytes, const std::vector<double> &ts) {
  std::vector<double> s = ts;
  std::sort(s.begin(), s.end());
  std::printf("{\"engine\": \"%s\", \"file\": \"%s\", \"bytes\": %zu, \"seconds\": %.9f, \"samples\": [", engine, file,
              bytes, s[s.size() / 2]);
  for (size_t i = 0; i < ts.size(); ++i) std::printf("%s%.9f", i ? ", " : "", ts[i]);
  std::printf("]}\n");
}

int main(int argc, char **argv) {
  if (argc < 4) die("usage: bench <libbase.so> <json.cbor> <file.json>...");
  void *lib = dlopen(argv[1], RTLD_NOW);
  if (!lib) die(dlerror());
  auto load = (void *(*)(const uint8_t *, size_t, char *, size_t))dlsym(lib, "base_driver_load");
  auto function = (BaseFunction(*)(const void *, const uint8_t *, size_t))dlsym(lib, "base_driver_function");
  std::string art = read(argv[2]);
  char err[512];
  void *drv = load((const uint8_t *)art.data(), art.size(), err, sizeof err);
  if (!drv) die(err);
  BaseFunction parse = function(drv, (const uint8_t *)"parse", 5), sum = function(drv, (const uint8_t *)"sum", 3);
  if (!parse || !sum) die("the artifact has no parse or sum");

  simdjson::dom::parser parser;
  for (int a = 3; a < argc; ++a) {
    const char *path = argv[a];
    const char *name = std::strrchr(path, '/') ? std::strrchr(path, '/') + 1 : path;
    std::string text = read(path);
    simdjson::padded_string padded(text);
    // the answers, compared
    int64_t st = parse((const uint8_t *)text.data(), text.size(), nullptr, 0);
    if (st != 0) {
      std::fprintf(stderr, "base refused %s at byte %lld\n", name, (long long)st - 1);
      return 1;
    }
    uint64_t hb = 0;
    sum(nullptr, 0, (uint8_t *)&hb, 8);
    Sum hs;
    hs.value(parser.parse(padded).value());
    if (hb != hs.h) {
      std::fprintf(stderr, "%s: base's checksum %016llx, simdjson's %016llx\n", name, (unsigned long long)hb,
                   (unsigned long long)hs.h);
      return 1;
    }
    // the times
    std::vector<double> ts;
    for (int r = -1; r < REPS; ++r) {
      double t = now();
      auto doc = parser.parse(padded);
      double dt = now() - t;
      if (doc.error()) die("simdjson refused a file");
      if (r >= 0) ts.push_back(dt);
    }
    emit("simdjson", name, text.size(), ts);
    ts.clear();
    for (int r = -1; r < REPS; ++r) {
      double t = now();
      int64_t s = parse((const uint8_t *)text.data(), text.size(), nullptr, 0);
      double dt = now() - t;
      if (s != 0) die("base refused a file it parsed before");
      if (r >= 0) ts.push_back(dt);
    }
    emit("base", name, text.size(), ts);
  }
  return 0;
}
