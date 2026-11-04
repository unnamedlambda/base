//! Each instruction computes what CLIF says: programs using instructions the
//! corpora do not, with the values checked.

use crate::common::*;

/// Runs `insts` and answers the `n` bytes they left at arena offset 2000.
fn compute(insts: Vec<Inst>, n: usize) -> Vec<u8> {
    let mut body = insts;
    body.push(ret());
    let d = driver(main_of(body));
    call_with(&d, "main", &[], &mut []);
    read(&d, 2000, n)
}

fn i64s(bytes: &[u8]) -> Vec<i64> {
    bytes.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect()
}

#[test]
fn instr_integer_arithmetic() {
    // udiv, ineg, band, band_not, bor, bxor
    let bytes = compute(
        vec![
            // A negative dividend: unsigned division of -100 is a huge
            // quotient, signed division would be -14.
            iconst64(v(1), -100),
            iconst64(v(2), 7),
            udiv(v(3), v(1), v(2)),
            store(v(3), v(0), 2000),
            ineg(v(4), v(2)),
            store(v(4), v(0), 2008),
            iconst64(v(5), 0xF0),
            iconst64(v(6), 0x3C),
            band(v(7), v(5), v(6)),
            store(v(7), v(0), 2016),
            band_not(v(8), v(5), v(6)),
            store(v(8), v(0), 2024),
            bor(v(9), v(5), v(6)),
            store(v(9), v(0), 2032),
            bxor(v(10), v(5), v(6)),
            store(v(10), v(0), 2040),
        ],
        48,
    );
    assert_eq!(
        i64s(&bytes),
        vec![
            ((-100i64) as u64 / 7) as i64, // unsigned, not -14
            -7,           // -(7)
            0x30,         // F0 & 3C
            0xC0,         // F0 & !3C
            0xFC,         // F0 | 3C
            0xCC,         // F0 ^ 3C
        ]
    );
}

#[test]
fn instr_bit_counting_and_select() {
    // ctz, popcnt, select, bitselect
    let bytes = compute(
        vec![
            // 0b1_0000 has one set bit and four trailing zeros, so a swap of
            // the two would show.
            iconst64(v(1), 0b1_0000),
            ctz(v(2), v(1)),
            store(v(2), v(0), 2000),
            popcnt(v(3), v(1)),
            store(v(3), v(0), 2008),
            // select picks by a condition, bitselect picks by a mask
            iconst64(v(4), 111),
            iconst64(v(5), 222),
            iconst64(v(6), 1),
            iconst64(v(7), 2),
            icmp(v(8), "ult", v(6), v(7)),
            select(v(9), v(8), v(4), v(5)),
            store(v(9), v(0), 2016),
            iconst64(v(10), 0xFF00),
            bitselect(v(11), v(10), v(4), v(5)),
            store(v(11), v(0), 2024),
        ],
        32,
    );
    let got = i64s(&bytes);
    assert_eq!(got[0], 4, "ctz(0b1_0000)");
    assert_eq!(got[1], 1, "popcnt(0b1_0000)");
    assert_eq!(got[2], 111, "select(1 < 2, 111, 222)");
    assert_eq!(got[3], (111 & 0xFF00) | (222 & !0xFF00), "bitselect");
}

#[test]
fn instr_width_conversions() {
    // ireduce32, uextend64, sextend64, istore8, an aligned store
    let bytes = compute(
        vec![
            // sign extension differs from zero extension for a negative i32
            iconst32(v(1), -5),
            sextend64(v(2), v(1)),
            store(v(2), v(0), 2000),
            uextend64(v(3), v(1)),
            store(v(3), v(0), 2008),
            // narrowing keeps the low 32 bits
            iconst64(v(4), 0x1_0000_002A),
            ireduce32(v(5), v(4)),
            uextend64(v(6), v(5)),
            store(v(6), v(0), 2016),
            // istore8 writes one byte; an aligned store states its alignment
            iconst64(v(7), 0xAB),
            istore8(v(7), v(0), 2024),
            iconst64(v(8), 77),
            store_aligned(v(8), v(0), 2032),
        ],
        40,
    );
    let got = i64s(&bytes);
    assert_eq!(got[0], -5, "sextend64(-5i32)");
    assert_eq!(got[1], 0xFFFF_FFFB, "uextend64(-5i32)");
    assert_eq!(got[2], 0x2A, "ireduce32 keeps the low word");
    assert_eq!(got[3] & 0xFF, 0xAB, "istore8");
    assert_eq!(got[4], 77, "aligned store");
}

#[test]
fn instr_float_arithmetic() {
    // f32const, f64const, fadd, fsub, fmul, fneg, fmax, fmin, fpromote
    let bytes = compute(
        vec![
            f64const(v(1), 3.5),
            f64const(v(2), 1.25),
            fadd(v(3), v(1), v(2)),
            store(v(3), v(0), 2000),
            fsub(v(4), v(1), v(2)),
            store(v(4), v(0), 2008),
            fmul(v(5), v(1), v(2)),
            store(v(5), v(0), 2016),
            fneg(v(6), v(1)),
            store(v(6), v(0), 2024),
            fmax(v(7), v(1), v(2)),
            store(v(7), v(0), 2032),
            fmin(v(8), v(1), v(2)),
            store(v(8), v(0), 2040),
            // f32 -> f64 keeps the value
            f32const(v(9), 2.5),
            fpromote(v(10), v(9)),
            store(v(10), v(0), 2048),
        ],
        56,
    );
    let got: Vec<f64> = bytes
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(got, vec![4.75, 2.25, 4.375, -3.5, 3.5, 1.25, 2.5]);
}

#[test]
fn instr_float_conversions_and_compare() {
    // fcvt_from_sint, fcvt_to_uint, bitcast, fcmp
    let bytes = compute(
        vec![
            iconst64(v(1), 9),
            fcvt_from_sint(v(2), F64, v(1)),
            f64const(v(3), 0.5),
            fmul(v(4), v(2), v(3)),
            store(v(4), v(0), 2000), // 4.5
            fcvt_to_uint(v(5), I64, v(4)),
            store(v(5), v(0), 2008), // 4, truncated
            // bitcast reinterprets rather than converts
            bitcast(v(6), I64, v(3)),
            store(v(6), v(0), 2016),
            // fcmp yields a one-byte flag, widened here so it can be read back
            fcmp(v(7), "gt", v(2), v(3)),
            uextend64(v(70), v(7)),
            store(v(70), v(0), 2024),
            fcmp(v(8), "lt", v(2), v(3)),
            uextend64(v(80), v(8)),
            store(v(80), v(0), 2032),
        ],
        40,
    );
    assert_eq!(f64::from_le_bytes(bytes[0..8].try_into().unwrap()), 4.5);
    let got = i64s(&bytes);
    assert_eq!(got[1], 4, "fcvt_to_uint truncates");
    assert_eq!(got[2], 0.5f64.to_bits() as i64, "bitcast is a reinterpretation");
    assert_eq!(got[3], 1, "9.0 > 0.5");
    assert_eq!(got[4], 0, "9.0 < 0.5 is false");
}

#[test]
fn instr_vector_lanes() {
    // splat, extractlane, vhigh_bits
    let bytes = compute(
        vec![
            // Four distinct lanes staged in memory, so the lane index is what
            // decides the answer. Bits: [1.0, 2.0] then [3.0, 4.0].
            iconst64(v(1), 0x4000_0000_3F80_0000u64 as i64),
            store(v(1), v(0), 2200),
            iconst64(v(2), 0x4080_0000_4040_0000u64 as i64),
            store(v(2), v(0), 2208),
            load_aligned(v(3), F32X4, v(0), 2200),
            extractlane(v(4), v(3), 2),
            fpromote(v(5), v(4)),
            store(v(5), v(0), 2000),
            // and a splat really does fill every lane
            f32const(v(20), 1.5),
            splat(v(21), F32X4, v(20)),
            extractlane(v(22), v(21), 3),
            fpromote(v(23), v(22)),
            store(v(23), v(0), 2016),
            // vhigh_bits gathers the sign bit of each byte lane: sixteen 0xFF
            // bytes staged in memory, read back as one vector.
            iconst64(v(6), -1),
            store(v(6), v(0), 2100),
            store(v(6), v(0), 2108),
            load_aligned(v(7), I8X16, v(0), 2100),
            vhigh_bits(v(8), v(7)),
            uextend64(v(9), v(8)),
            store(v(9), v(0), 2008),
        ],
        24,
    );
    assert_eq!(
        f64::from_le_bytes(bytes[0..8].try_into().unwrap()),
        3.0,
        "lane 2 of [1, 2, 3, 4]"
    );
    assert_eq!(i64s(&bytes)[1], 0xFFFF, "every one of the sixteen lanes is negative");
    assert_eq!(
        f64::from_le_bytes(bytes[16..24].try_into().unwrap()),
        1.5,
        "splat fills lane 3 too"
    );
}
