use serde::{Deserialize, Serialize};

pub mod clif;

/// Bytes the memory starts with at one address.
///
/// Memory is zero everywhere a segment does not cover it, which is what keeps
/// an artifact from shipping the zeros: across the artifacts in this
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

/// The board's control, as data: the functions a host compiles, the memory
/// they run in, and the bytes that memory starts with.
///
/// This is the whole of the wire format. A host calls a function by its
/// [`clif::Function::entry_name`]; what the bytes it reads and writes mean is
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
/// - a newtype is its content, `None` is null, and segment bytes are a byte
///   string.
///
/// [`Artifact::to_bytes`] writes exactly this, so one file has one encoding:
/// the build refuses a generated artifact that does not re-encode to itself.
///
/// Unknown fields are refused rather than skipped. `data` defaults to empty, so
/// a writer left behind by a change to this struct would otherwise produce an
/// artifact that parses and starts from memory it believes it filled — which is
/// a program reading zeros, not a failure anyone would trace back to here.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Artifact {
    /// Compiled as a unit. A function's `u0:N` index is its position here.
    pub functions: Vec<clif::Function>,
    /// How much the program needs, which the runtime allocates and cannot
    /// check: nothing here says what the bodies go on to address, so an
    /// artifact asking for too little is a program reading its own zeros.
    /// What the generator proved its layout fits in is what belongs here.
    ///
    /// Sizes and addresses are 64-bit whatever the host: the artifact is the
    /// same file everywhere, and a host that cannot hold it says so when it
    /// loads it.
    pub required_memory: u64,
    /// In ascending order of address and non-overlapping, which is what lets
    /// the runtime lay them over zeroed memory in one pass.
    #[serde(default)]
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

    /// The `len` bytes the memory starts with at `offset`, zero-filled where no
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

    /// Put `bytes` at `offset` of the memory the program starts from — what a
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
        Artifact { functions: vec![], required_memory: 64, data }
    }

    fn hex(parts: &[&str]) -> Vec<u8> {
        let s: String = parts.concat();
        (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect()
    }

    /// One function touching each shape the profile has to spell: an optional
    /// name, unit and tuple variants, a newtype variant holding null, a nested
    /// struct, a tuple, and integers no double holds exactly.
    fn sample() -> Artifact {
        use clif::*;
        let (i64_, v) = (ClifTy::I64, Val);
        Artifact {
            functions: vec![Function {
                entry_name: Some("main".into()),
                callees: vec![Callee::Import("cl_x".into()), Callee::Local(1)],
                blocks: vec![Block {
                    reference: BlockRef(0),
                    params: vec![(v(0), i64_)],
                    insts: vec![
                        Inst::Iconst(v(1), i64_, i64::MIN),
                        Inst::Iconst(v(2), i64_, i64::MAX),
                        Inst::Fconst(v(3), ClifTy::F64, u64::MAX),
                        Inst::Load(
                            v(4),
                            LoadOp { kind: LoadKind::Uload8, ty: ClifTy::I32, notrap_aligned: false },
                            v(0),
                            0,
                        ),
                        Inst::Call(None, FnRef(0), vec![v(0)]),
                        Inst::Ret(None),
                    ],
                }],
            }],
            required_memory: 1 << 40,
            data: vec![Segment { offset: 3, bytes: vec![0xff, 0x00, 0x01] }],
        }
    }

    /// The profile, byte for byte. The expected bytes come from an encoder
    /// written separately from minicbor, and `AlgorithmLib.Core` checks that
    /// the Lean writer produces the same bytes for the same value.
    #[test]
    fn an_artifact_encodes_in_the_profile() {
        let expected = hex(&[
            "a36966756e6374696f6e7381a36a656e7472795f6e616d65646d61696e6763616c6c65657382a166",
            "496d706f727464636c5f78a1654c6f63616c0166626c6f636b7381a3697265666572656e63650066",
            "706172616d738182006349363465696e73747386a16649636f6e73748301634936343b7fffffffff",
            "ffffffa16649636f6e73748302634936341b7fffffffffffffffa16646636f6e7374830363463634",
            "1bffffffffffffffffa1644c6f61648404a3646b696e6466556c6f616438627479634933326e6e6f",
            "747261705f616c69676e6564f40000a16443616c6c83f6008100a163526574f66f72657175697265",
            "645f6d656d6f72791b0000010000000000646461746181a2666f66667365740365627974657343ff",
            "0001",
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
        assert_eq!(insts[0], clif::Inst::Iconst(clif::Val(1), clif::ClifTy::I64, i64::MIN));
        assert_eq!(insts[1], clif::Inst::Iconst(clif::Val(2), clif::ClifTy::I64, i64::MAX));
        assert_eq!(insts[2], clif::Inst::Fconst(clif::Val(3), clif::ClifTy::F64, u64::MAX));
        assert_eq!((i64::MAX - 1) as f64, i64::MAX as f64, "a double cannot tell these apart");
    }

    /// Instructions are named, not numbered, so a reader that does not know one
    /// says which — rather than reading it as whichever instruction has that
    /// number, which is what a positional encoding does.
    #[test]
    fn an_instruction_the_reader_does_not_know_is_refused_by_name() {
        let bytes = sample().to_bytes();
        let known = b"Iconst";
        let at = bytes.windows(known.len()).position(|w| w == known).unwrap();
        let mut renamed = bytes.clone();
        renamed[at..at + known.len()].copy_from_slice(b"Iconsx");
        let err = Artifact::from_bytes(&renamed).unwrap_err();
        assert!(err.contains("Iconsx"), "{err}");
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
