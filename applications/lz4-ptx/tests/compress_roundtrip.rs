use arrow_array::Int64Array;
use base::{Artifact, Base};

const WARP_COMP: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/Lz4CompAlgorithm/lz4_comp_warpdsl.bin"));
const WARP_COMP64: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/Lz4CompAlgorithm/lz4_comp_warpdsl64.bin"));

// Must match the shipped kernels' baked geometry (Lz4CompAlgorithm: blkLog 15/16,
// corpusBytes 209_715_200 => 6400 blocks of 32 KiB, or 3200 of 64 KiB).
const BLOCK: usize = 32768;
const NUM_BLK: usize = 6400;
const BLOCK64: usize = 65536;
const NUM_BLK64: usize = 3200;

/// Deterministic, LZ4-compressible corpus: literal runs interleaved with
/// in-window back-references (matches), giving a realistic ~2:1 ratio that
/// exercises the warp matchfinder. Matches stay within a 4 KiB window, so every
/// independent 32 KiB block is self-compressible.
fn gen_corpus(total: usize) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::with_capacity(total);
    let mut s: u32 = 0x9e37_79b9;
    let mut rng = move || {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        s
    };
    while out.len() < total {
        let r = rng();
        if (r & 3) != 0 && out.len() >= 8 {
            // back-reference (a match): copy a run from the recent window
            let win = out.len().min(4096);
            let dist = 1 + (rng() as usize) % win;
            let len = 4 + (r as usize >> 2) % 60; // 4..64
            let start = out.len() - dist;
            for k in 0..len {
                if out.len() >= total {
                    break;
                }
                out.push(out[start + (k % dist)]); // overlap-safe (RLE-style)
            }
        } else {
            // literal run of a single byte (still a within-block match source)
            let len = 2 + (r as usize >> 4) % 10;
            let b = (r >> 24) as u8;
            for _ in 0..len {
                if out.len() >= total {
                    break;
                }
                out.push(b);
            }
        }
    }
    out.truncate(total);
    out
}

fn check_artifact(artifact: &[u8], block: usize, num_blk: usize) {
    let original = gen_corpus(num_blk * block);

    let art = Artifact::from_bytes(artifact);
    let mut base = Base::new(art.setup).expect("compile");
    let col = |b: &arrow_array::RecordBatch, i: usize| {
        b.column(i).as_any().downcast_ref::<Int64Array>().unwrap().value(0)
    };

    let len_off = block + block / 16 + 256;
    let out_stride = len_off + 8;
    let mut out = vec![0u8; num_blk * out_stride];
    let batches = base.execute_into(&art.main, &original, &mut out).expect("run");
    assert_eq!(col(&batches[0], 5) as usize, len_off, "len_off mismatch");
    assert_eq!(col(&batches[0], 6) as usize, num_blk, "num_blk mismatch");

    // Negative control (self-check): with LZ4_NEG_CONTROL set, corrupt block 0's
    // expected bytes so a genuine byte-exact check MUST fail — proves this test
    // can't silently pass. No-op on normal runs.
    let neg = std::env::var("LZ4_NEG_CONTROL").is_ok();

    let mut total_comp = 0usize;
    for i in 0..num_blk {
        let bo = i * out_stride;
        let clen =
            u32::from_le_bytes(out[bo + len_off..bo + len_off + 4].try_into().unwrap()) as usize;
        assert!(clen > 0 && clen <= len_off, "block {i}: bad compLen {clen}");
        total_comp += clen;
        let decomp = lz4_flex::block::decompress(&out[bo..bo + clen], block)
            .unwrap_or_else(|e| panic!("block {i}: decode failed: {e:?}"));
        let exp = &original[i * block..(i + 1) * block];
        if neg && i == 0 {
            let mut c = exp.to_vec();
            c[block / 2] ^= 0xff;
            assert_eq!(&decomp[..], &c[..], "block {i}: did not decode to the original");
        } else {
            assert_eq!(&decomp[..], exp, "block {i}: did not decode to the original");
        }
    }
    let ratio = (num_blk * block) as f64 / total_comp as f64;
    eprintln!(
        "validated {num_blk} blocks of {block} B byte-exact on GPU, compression ratio {ratio:.3}:1"
    );
    assert!(ratio > 1.30, "compression ratio regressed: {ratio:.3}");
}

#[test]
fn warp_compress_blocks_decode_byte_exact() {
    check_artifact(WARP_COMP, BLOCK, NUM_BLK);
}

#[test]
fn warp_compress64_blocks_decode_byte_exact() {
    check_artifact(WARP_COMP64, BLOCK64, NUM_BLK64);
}
