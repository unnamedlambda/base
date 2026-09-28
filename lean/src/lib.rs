//! The artifacts this Lake package generates, one `pub static` each, named after
//! the artifact in capitals:
//!
//! ```ignore
//! let artifact = base::Artifact::from_bytes(lean_artifacts::BYTE_SCRUB)?;
//! ```
//!
//! Depending on this crate is all a consumer needs: its build script runs
//! `lake`, which rebuilds only the generators whose sources changed, and cargo
//! reruns it only when a file those generators import does. [`by_name`] looks
//! one up at run time, and [`DIR`] is where they are, with an artifact's side
//! data in `DIR/<name>/`.

include!(concat!(env!("OUT_DIR"), "/artifacts.rs"));
