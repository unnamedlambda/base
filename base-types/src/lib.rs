use serde::{Deserialize, Serialize};

pub mod clif;

/// Bytes the arena starts with at one address.
///
/// The arena is zero everywhere a segment does not cover it, which is what
/// keeps an artifact from shipping the zeros: across the artifacts in this
/// repository the images are 29.7 MB, of which 4.9 MB is not zero.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Segment {
    pub offset: u64,
    #[serde(with = "serde_bytes")]
    pub bytes: Vec<u8>,
}

impl Segment {
    /// The first address past the segment. Saturating, so a segment no memory
    /// could hold compares past every address rather than wrapping around.
    fn end(&self) -> u64 {
        self.offset.saturating_add(self.bytes.len() as u64)
    }
}

/// The board's control, as data: one Cranelift module. The primitives it
/// calls, the functions it defines, and the one piece of data it declares —
/// the arena they run in — with the bytes the arena starts with.
///
/// This is the whole of the wire format. A host calls a function by its
/// [`clif::Function::name`]; what the bytes it reads and writes mean is
/// between the program and the host, and lives in those bytes when it needs to.
///
/// # Encoding
///
/// CBOR (RFC 8949), the same file for every host, in one profile:
///
/// - lengths are definite, and every head is the shortest that holds its value;
/// - a struct is a map whose text keys are its fields in declaration order —
///   deliberately not the key sorting of RFC 8949 §4.2.1, so the field order
///   here is part of the format;
/// - an enum is externally tagged: a unit variant is its name as text, any
///   other variant a one-entry map from its name to its value, with a tuple
///   variant's fields as an array;
/// - a newtype is its content, `None` is null, a tuple is an array, and segment
///   bytes are a byte string.
///
/// [`Artifact::to_bytes`] writes exactly this, so one file has one encoding:
/// the build refuses a generated artifact that does not re-encode to itself.
///
/// Unknown fields are refused rather than skipped, and every field is written,
/// so a writer left behind by a change to this struct fails to load rather
/// than producing a program that starts from an arena it believes it filled.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Artifact {
    /// The C functions the program calls, each declared once; a call names
    /// one by its position here.
    pub primitives: Vec<clif::Primitive>,
    /// Compiled as a unit. A call names one by its position here.
    pub functions: Vec<clif::Function>,
    /// The size of the arena, which the engine allocates exactly and cannot
    /// check: nothing here says what the bodies go on to address, so an
    /// artifact asking for too little is a program reading its own zeros.
    /// What the generator proved its layout fits in is what belongs here.
    ///
    /// The arena is page-aligned, so an address a program keeps aligned
    /// relative to its start is aligned in fact, which an `aligned` access
    /// states. Sizes and addresses are 64-bit whatever the host: the artifact
    /// is the same file everywhere, and a host that cannot hold it says so
    /// when it loads it.
    pub arena_size: u64,
    /// In ascending order of address and not overlapping, which is what lets
    /// the engine lay them over the zeroed arena in one pass, and what gives an
    /// arena one spelling; a segment out of order, overlapping another, or
    /// past the arena is refused at load.
    pub data: Vec<Segment>,
}

impl Artifact {
    /// Decode an artifact, refusing trailing bytes after it.
    pub fn from_bytes(bytes: &[u8]) -> Result<Artifact, String> {
        let mut de = minicbor_serde::Deserializer::new(bytes);
        let artifact = Artifact::deserialize(&mut de).map_err(|e| format!("not an artifact: {e}"))?;
        let rest = bytes.len() - de.decoder().position();
        if rest != 0 {
            return Err(format!("not an artifact: {rest} bytes follow it"));
        }
        Ok(artifact)
    }

    /// Encode an artifact in the profile described above.
    pub fn to_bytes(&self) -> Vec<u8> {
        minicbor_serde::to_vec(self).expect("an artifact has nothing serde cannot write")
    }

    /// The `len` bytes the arena starts with at `offset`, zero-filled where no
    /// segment covers them.
    pub fn read(&self, offset: u64, len: usize) -> Vec<u8> {
        let mut out = vec![0u8; len];
        let end = offset.saturating_add(len as u64);
        for s in &self.data {
            let from = s.offset.max(offset);
            let to = s.end().min(end);
            if from < to {
                let (a, b) = ((from - offset) as usize, (to - offset) as usize);
                let (c, d) = ((from - s.offset) as usize, (to - s.offset) as usize);
                out[a..b].copy_from_slice(&s.bytes[c..d]);
            }
        }
        out
    }

    /// Put `bytes` at `offset` of the arena the program starts from — what a
    /// host does when a path or a length is known only when it runs.
    ///
    /// Segments the write touches or abuts are merged with it, so the result is
    /// still ordered and non-overlapping.
    pub fn write(&mut self, offset: u64, bytes: &[u8]) {
        let (mut lo, mut hi) = (offset, offset.saturating_add(bytes.len() as u64));
        let mut touched: Vec<Segment> = Vec::new();
        self.data.retain(|s| {
            let overlaps = s.offset <= hi && lo <= s.end();
            if overlaps {
                lo = lo.min(s.offset);
                hi = hi.max(s.end());
                touched.push(s.clone());
            }
            !overlaps
        });
        let mut merged = vec![0u8; (hi - lo) as usize];
        for s in touched {
            let at = (s.offset - lo) as usize;
            merged[at..at + s.bytes.len()].copy_from_slice(&s.bytes);
        }
        let at = (offset - lo) as usize;
        merged[at..at + bytes.len()].copy_from_slice(bytes);
        self.data.push(Segment { offset: lo, bytes: merged });
        self.data.sort_by_key(|s| s.offset);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn artifact(data: Vec<Segment>) -> Artifact {
        Artifact { primitives: vec![], functions: vec![], arena_size: 64, data }
    }

    fn hex(parts: &[&str]) -> Vec<u8> {
        let s: String = parts.concat();
        (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect()
    }

    /// One function touching each shape the profile has to spell: an optional
    /// name, unit, newtype and tuple variants, null, a nested struct, a tuple,
    /// and integers no double holds exactly.
    fn sample() -> Artifact {
        use clif::*;
        let v = Val;
        Artifact {
            primitives: vec![Primitive {
                lib: "base".into(),
                symbol: "cl_sinf".into(),
                signature: Signature {
                    convention: Convention::Host,
                    params: vec![ClifTy::F32.into()],
                    returns: vec![ClifTy::F32.into()],
                },
            }],
            functions: vec![Function {
                name: Some("main".into()),
                signature: Signature::base_function(),
                stack_slots: vec![],
                blocks: vec![Block {
                    reference: BlockRef(0),
                    params: vec![(v(0), ClifTy::I64)],
                    insts: vec![
                        Inst::UnaryImm(vec![v(1)], "iconst".into(), Some(ClifTy::I64), i64::MIN),
                        Inst::UnaryImm(vec![v(2)], "iconst".into(), Some(ClifTy::I64), i64::MAX),
                        Inst::UnaryIeee64(vec![v(3)], "f64const".into(), None, u64::MAX),
                        Inst::Load(vec![v(4)], "uload8".into(), Some(ClifTy::I32), vec![], v(0), 0),
                        // A primitive, whose result nothing reads, and a function.
                        Inst::Call(vec![v(6)], "call".into(), None, Callee::Primitive(0), vec![v(0)]),
                        Inst::Call(vec![], "call".into(), None, Callee::Function(1), vec![v(0)]),
                        // An indirect one, at a signature and a convention.
                        Inst::CallIndirect(
                            vec![v(5)],
                            "call_indirect".into(),
                            None,
                            Signature {
                                convention: Convention::Arch,
                                params: vec![ClifTy::I64.into(); 4],
                                returns: vec![ClifTy::I64.into()],
                            },
                            v(0),
                            vec![v(0); 4],
                        ),
                        Inst::MultiAry(vec![], "return".into(), None, vec![]),
                    ],
                }],
            }],
            arena_size: 1 << 40,
            data: vec![Segment { offset: 3, bytes: vec![0xff, 0x00, 0x01] }],
        }
    }

    /// The profile, byte for byte. `AlgorithmLib.Core` checks that the Lean
    /// writer, written separately from minicbor, produces the same bytes for
    /// the same value, so a change to either side shows here or there.
    #[test]
    fn an_artifact_encodes_in_the_profile() {
        let expected = hex(&[
            "a46a7072696d69746976657381a3636c696264626173656673796d626f6c67636c5f73696e666973",
            "69676e6174757265a36a636f6e76656e74696f6e64486f737466706172616d738182636633328067",
            "72657475726e73818263663332806966756e6374696f6e7381a4646e616d65646d61696e69736967",
            "6e6174757265a36a636f6e76656e74696f6e64486f737466706172616d7384826369363480826369",
            "3634808263693634808263693634806772657475726e73818263693634806b737461636b5f736c6f",
            "74738066626c6f636b7381a3697265666572656e63650066706172616d738182006369363465696e",
            "73747388a168556e617279496d6d8481016669636f6e7374636936343b7fffffffffffffffa16855",
            "6e617279496d6d8481026669636f6e7374636936341b7fffffffffffffffa16b556e617279496565",
            "65363484810368663634636f6e7374f61bffffffffffffffffa1644c6f616486810466756c6f6164",
            "3863693332800000a16443616c6c8581066463616c6cf6a1695072696d6974697665008100a16443",
            "616c6c85806463616c6cf6a16846756e6374696f6e018100a16c43616c6c496e6469726563748681",
            "056d63616c6c5f696e646972656374f6a36a636f6e76656e74696f6e644172636866706172616d73",
            "848263693634808263693634808263693634808263693634806772657475726e7381826369363480",
            "008400000000a1684d756c746941727984806672657475726ef6806a6172656e615f73697a651b00",
            "00010000000000646461746181a2666f66667365740365627974657343ff0001",
        ]);
        assert_eq!(sample().to_bytes(), expected);
        assert_eq!(Artifact::from_bytes(&expected).unwrap(), sample());
    }

    /// Integers come back exactly. JSON read through doubles — every
    /// JavaScript host — turns `i64::MAX` into 9223372036854775808 and the
    /// all-ones float pattern into a different pattern; CBOR carries them as
    /// integers, so there is nothing to round.
    #[test]
    fn integers_past_two_to_the_53_come_back_exactly() {
        let back = Artifact::from_bytes(&sample().to_bytes()).unwrap();
        let insts = &back.functions[0].blocks[0].insts;
        let iconst = |d, k| clif::Inst::UnaryImm(vec![clif::Val(d)], "iconst".into(), Some(clif::ClifTy::I64), k);
        assert_eq!(insts[0], iconst(1, i64::MIN));
        assert_eq!(insts[1], iconst(2, i64::MAX));
        assert_eq!(insts[2], clif::Inst::UnaryIeee64(vec![clif::Val(3)], "f64const".into(), None, u64::MAX));
        assert_eq!((i64::MAX - 1) as f64, i64::MAX as f64, "a double cannot tell these apart");
    }

    /// Instructions are named, not numbered, so a reader that does not know one
    /// says which — rather than reading it as whichever instruction has that
    /// number, which is what a positional encoding does.
    #[test]
    fn an_instruction_the_reader_does_not_know_is_refused_by_name() {
        let bytes = sample().to_bytes();
        let known = b"UnaryImm";
        let at = bytes.windows(known.len()).position(|w| w == known).unwrap();
        let mut renamed = bytes.clone();
        renamed[at..at + known.len()].copy_from_slice(b"UnaryImx");
        let err = Artifact::from_bytes(&renamed).unwrap_err();
        assert!(err.contains("UnaryImx"), "{err}");
    }

    /// A file is one artifact and nothing else, and JSON is not an artifact.
    #[test]
    fn trailing_bytes_and_json_are_refused() {
        let mut bytes = sample().to_bytes();
        bytes.push(0);
        assert_eq!(Artifact::from_bytes(&bytes).unwrap_err(), "not an artifact: 1 bytes follow it");
        let json = serde_json::to_vec(&sample()).unwrap();
        assert!(Artifact::from_bytes(&json).is_err());
    }

    /// Reading spans segments and the zeros between them.
    #[test]
    fn read_zero_fills_what_no_segment_covers() {
        let a = artifact(vec![
            Segment { offset: 0, bytes: vec![1, 2] },
            Segment { offset: 6, bytes: vec![9] },
        ]);
        assert_eq!(a.read(0, 8), vec![1, 2, 0, 0, 0, 0, 9, 0]);
        assert_eq!(a.read(2, 2), vec![0, 0]);
    }

    /// A write that lands in the zeros becomes a segment of its own; one that
    /// touches existing segments joins them, and what it does not cover keeps
    /// the bytes that were there.
    #[test]
    fn writing_keeps_segments_ordered_and_disjoint() {
        let mut a = artifact(vec![
            Segment { offset: 0, bytes: vec![1, 2] },
            Segment { offset: 6, bytes: vec![9] },
        ]);
        a.write(20, &[7, 7]);
        assert_eq!(a.data.len(), 3, "a write among the zeros is its own segment");
        assert_eq!(a.read(20, 2), vec![7, 7]);

        a.write(2, &[3, 4, 5]);
        assert_eq!(a.data.len(), 3, "it joined the segment it abuts, not the one a zero away");
        assert_eq!(a.read(0, 8), vec![1, 2, 3, 4, 5, 0, 9, 0]);
        assert!(a.data.windows(2).all(|w| w[0].end() <= w[1].offset), "ordered and disjoint");
    }

    /// Overwriting inside one segment leaves the rest of it alone.
    #[test]
    fn writing_inside_a_segment_replaces_only_those_bytes() {
        let mut a = artifact(vec![Segment { offset: 4, bytes: vec![1, 2, 3, 4] }]);
        a.write(5, &[8]);
        assert_eq!(a.data.len(), 1);
        assert_eq!(a.read(4, 4), vec![1, 8, 3, 4]);
    }
}
