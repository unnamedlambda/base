// CUB's DeviceRadixSort::SortPairs on (u32 key, u32 value) pairs, as the CUDA
// toolkit ships it: the frontier for bench.py's sort.
//
//   cub_sort <keys.bin> <n> <reps> <out.bin>
//
// Reads n u32 keys, pairs them with values 0 … n-1, sorts on the device
// (temporary storage allocated once, outside the timing), prints the seconds
// of each of `reps` sorts, each between two events, a line each, and writes the sorted
// keys then values to out.bin for bench.py to check.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cub/device/device_radix_sort.cuh>

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  std::fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e)); std::exit(1); } } while (0)

int main(int argc, char **argv) {
  if (argc != 5) { std::fprintf(stderr, "usage: cub_sort keys.bin n reps out.bin\n"); return 2; }
  size_t n = std::strtoull(argv[2], nullptr, 10);
  int reps = std::atoi(argv[3]);
  std::vector<unsigned> keys(n), vals(n);
  FILE *f = std::fopen(argv[1], "rb");
  if (!f || std::fread(keys.data(), 4, n, f) != n) { std::fprintf(stderr, "cannot read keys\n"); return 1; }
  std::fclose(f);
  for (size_t i = 0; i < n; ++i) vals[i] = (unsigned)i;

  unsigned *kin, *kout, *vin, *vout;
  CHECK(cudaMalloc(&kin, n * 4)); CHECK(cudaMalloc(&kout, n * 4));
  CHECK(cudaMalloc(&vin, n * 4)); CHECK(cudaMalloc(&vout, n * 4));
  CHECK(cudaMemcpy(kin, keys.data(), n * 4, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(vin, vals.data(), n * 4, cudaMemcpyHostToDevice));
  size_t tmpBytes = 0;
  CHECK(cub::DeviceRadixSort::SortPairs(nullptr, tmpBytes, kin, kout, vin, vout, (int)n));
  void *tmp;
  CHECK(cudaMalloc(&tmp, tmpBytes));

  cudaEvent_t a, b;
  CHECK(cudaEventCreate(&a)); CHECK(cudaEventCreate(&b));
  CHECK(cub::DeviceRadixSort::SortPairs(tmp, tmpBytes, kin, kout, vin, vout, (int)n));  // warm-up
  std::vector<float> ms(reps);
  for (int r = 0; r < reps; ++r) {
    CHECK(cudaEventRecord(a));
    CHECK(cub::DeviceRadixSort::SortPairs(tmp, tmpBytes, kin, kout, vin, vout, (int)n));
    CHECK(cudaEventRecord(b));
    CHECK(cudaEventSynchronize(b));
    CHECK(cudaEventElapsedTime(&ms[r], a, b));
  }
  for (int r = 0; r < reps; ++r) std::printf("%.9f\n", ms[r] / 1e3);

  CHECK(cudaMemcpy(keys.data(), kout, n * 4, cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(vals.data(), vout, n * 4, cudaMemcpyDeviceToHost));
  FILE *o = std::fopen(argv[4], "wb");
  std::fwrite(keys.data(), 4, n, o);
  std::fwrite(vals.data(), 4, n, o);
  std::fclose(o);
  return 0;
}
