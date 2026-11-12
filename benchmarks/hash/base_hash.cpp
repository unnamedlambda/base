/*
 * base's rapidhash (lean/algorithms/Bench/Hash.lean) in SMHasher3: the
 * artifact loaded through base's C API at the hash's init, called through it
 * on every hash. The C API takes (input, length, output, output length) and
 * no seed, so the seed rides in the output word: base reads it there and
 * writes the hash over it. Its verification value is rapidhash v3's own, so
 * SMHasher3 reports it only if it is that function.
 *
 * BASE_LIB and BASE_HASH name libbase.so and the artifact (speed.py sets
 * them). Copied into SMHasher3's hashes/ by prepare.py.
 */
#include "Platform.h"
#include "Hashlib.h"

#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

typedef int64_t (* BaseFunction)( const uint8_t *, size_t, uint8_t *, size_t );

static BaseFunction base_rapidhash_fn;
static BaseFunction base_gxhash_fn;

static bool base_init( void ) {
    const char * lib = getenv("BASE_LIB");
    const char * art = getenv("BASE_HASH");
    if ((lib == NULL) || (art == NULL)) { return false; }
    void * h = dlopen(lib, RTLD_NOW);
    if (h == NULL) { return false; }
    auto load     = (void * (*)( const uint8_t *, size_t, char *, size_t ))dlsym(h, "base_driver_load");
    auto function = (BaseFunction (*)( const void *, const uint8_t *, size_t ))dlsym(h, "base_driver_function");
    if ((load == NULL) || (function == NULL)) { return false; }
    FILE * f = fopen(art, "rb");
    if (f == NULL) { return false; }
    std::vector<uint8_t> bytes;
    uint8_t buf[65536];
    size_t  n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) { bytes.insert(bytes.end(), buf, buf + n); }
    fclose(f);
    char  err[512];
    void * drv = load(bytes.data(), bytes.size(), err, sizeof(err));
    if (drv == NULL) { fprintf(stderr, "base: %s\n", err); return false; }
    base_rapidhash_fn = function(drv, (const uint8_t *)"rapidhash", 9);
    base_gxhash_fn    = function(drv, (const uint8_t *)"gxhash", 6);
    if ((base_rapidhash_fn == NULL) || (base_gxhash_fn == NULL)) { return false; }
    // gxhash needs AES and AVX: its status is 2 on a CPU without them
    uint64_t w = 0;
    return base_gxhash_fn((const uint8_t *)"", 0, (uint8_t *)&w, 8) == 0;
}

static void BaseRapidHash( const void * in, const size_t len, const seed_t seed, void * out ) {
    uint64_t w = (uint64_t)seed;
    base_rapidhash_fn((const uint8_t *)in, len, (uint8_t *)&w, 8);
    PUT_U64<false>(w, (uint8_t *)out, 0);
}

static void BaseGxHash64( const void * in, const size_t len, const seed_t seed, void * out ) {
    uint64_t w = (uint64_t)seed;
    base_gxhash_fn((const uint8_t *)in, len, (uint8_t *)&w, 8);
    PUT_U64<false>(w, (uint8_t *)out, 0);
}

REGISTER_FAMILY(base,
   $.src_url    = "https://github.com/Nicoshev/rapidhash",
   $.src_status = HashFamilyInfo::SRC_ACTIVE
 );

REGISTER_HASH(base_rapidhash,
   $.desc       = "rapidhash v3, 64-bit, as base's CLIF through its C API",
   $.hash_flags =
     0,
   $.impl_flags =
         FLAG_IMPL_MULTIPLY_64_128  |
         FLAG_IMPL_LICENSE_MIT,
   $.bits = 64,
   $.verification_LE = 0x1FDC65EE,
   $.verification_BE = 0xB2DB16B5,
   $.initfn          = base_init,
   $.hashfn_native   = BaseRapidHash,
   $.hashfn_bswap    = BaseRapidHash
 );

REGISTER_HASH(base_gxhash64,
   $.desc       = "gxhash, lower 64 bits, as base's own AES loop through its C API",
   $.hash_flags =
         FLAG_HASH_AES_BASED           |
         FLAG_HASH_ENDIAN_INDEPENDENT,
   $.impl_flags =
         FLAG_IMPL_READ_PAST_EOB       |
         FLAG_IMPL_LICENSE_MIT,
   $.bits = 64,
   $.verification_LE = 0x48F84240,
   $.verification_BE = 0x48F84240,
   $.initfn          = base_init,
   $.hashfn_native   = BaseGxHash64,
   $.hashfn_bswap    = BaseGxHash64
 );
