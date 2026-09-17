//! Generation of the Lean artifacts, and nothing else.
//!
//! This crate holds no artifacts. Its build script is the only caller of `lake`
//! in the workspace: it builds every generator declared in the algorithms
//! lakefile, runs each one, and writes the results to [`DIR`].
//!
//! Applications depend on this crate so that cargo runs the generation before
//! compiling them, then reach the bytes with `include_bytes!` against a path
//! under [`DIR`]. That keeps each artifact a separate file tracked by rustc,
//! rather than a constant every application would link.

/// Absolute path of the generated artifact tree, laid out as
/// `<module>/<artifact>.cbor`.
pub const DIR: &str = env!("LEAN_ARTIFACT_DIR");
