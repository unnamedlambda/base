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
/// This is the whole of the wire format. An entry point is a function index —
/// which function does what is the generator's knowledge, and stays with
/// whoever built the artifact.
///
/// Unknown fields are refused rather than skipped. `data` defaults to empty, so
/// a writer left behind by a change to this struct would otherwise produce an
/// artifact that parses and starts from memory it believes it filled — which is
/// a program reading zeros, not a failure anyone would trace back to here.
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct Artifact {
    /// Compiled as a unit. A function's `u0:N` index is its position here.
    pub functions: Vec<clif::Function>,
    /// Sizes and addresses are 64-bit whatever the host: the artifact is the
    /// same file everywhere, and a host that cannot hold it says so when it
    /// loads it.
    pub memory_size: u64,
    /// In ascending order of address and non-overlapping, which is what lets
    /// the runtime lay them over zeroed memory in one pass.
    #[serde(default)]
    pub data: Vec<Segment>,
}

impl Artifact {
    pub fn from_bytes(bytes: &[u8]) -> Artifact {
        bincode::deserialize(bytes).expect("failed to deserialize artifact")
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
        Artifact { functions: vec![], memory_size: 64, data }
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
