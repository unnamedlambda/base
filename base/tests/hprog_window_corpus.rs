//! Runs the window corpus: what `Sem.callFfi` says the window entry points do
//! without a display, against what `Lib.Window` does over the engine's window
//! library.
//!
//! The display variables are removed before the run, so the window library's
//! event loop does not start and `windowInit` stores null; the body then takes every refusal path and
//! the argument checks in front of them, leaving each result code in the output
//! buffer. The interpreter computed that buffer when the artifact was generated
//! (`lean/algorithms/Host/WindowCorpus.lean`); here the same artifact runs
//! through the JIT, and the two must agree byte for byte.

use base_types::Artifact;

use std::path::PathBuf;

/// Result codes, `i32`s from byte 0.
const WHAT: [&str; 8] = [
    "open with the null context init left",
    "open with width 0 refused",
    "open with an empty blit shader refused",
    "poll with a null context",
    "poll for a negative count refused",
    "poll into a null buffer refused",
    "present a negative buffer id refused",
    "present with a null context",
];

#[test]
fn interpreter_and_runtime_agree_without_a_display() {
    let dir = match std::env::var("BASE_HPROG_DIR") {
        Ok(d) => PathBuf::from(d),
        Err(_) => PathBuf::from(lean_artifacts::DIR),
    };
    let v: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join("hprog_window_corpus/expected.json")).expect("read corpus"),
    )
    .expect("corpus shape");
    let expected: Vec<u8> = v["expected"]
        .as_array()
        .expect("expected")
        .iter()
        .map(|b| b.as_u64().expect("byte") as u8)
        .collect();
    let a = Artifact::from_bytes(lean_artifacts::HPROG_WINDOW_CORPUS).expect("artifact shape");

    // This test binary is its own process and runs one test, so nothing else
    // reads the environment while it changes.
    for var in ["DISPLAY", "WAYLAND_DISPLAY", "WAYLAND_SOCKET"] {
        std::env::remove_var(var);
    }
    let mut b = base::Driver::load(a).expect("compile");
    let mut out = vec![0u8; expected.len()];
    b.execute("main", &[], &mut out).expect("execute");

    let mut bad = Vec::new();
    for (k, what) in WHAT.iter().enumerate() {
        let (got, want) = (&out[4 * k..4 * k + 4], &expected[4 * k..4 * k + 4]);
        if got != want {
            bad.push(format!(
                "  {what}: lean {} != runtime {}",
                i32::from_le_bytes(want.try_into().unwrap()),
                i32::from_le_bytes(got.try_into().unwrap())
            ));
        }
    }
    for (lo, hi, what) in [
        (64, 96, "events buffer left alone"),
        (128, 136, "slot after the first init"),
        (136, 144, "slot after the second init"),
    ] {
        if out[lo..hi] != expected[lo..hi] {
            bad.push(format!("  {what}: lean {:?} != runtime {:?}", &expected[lo..hi], &out[lo..hi]));
        }
    }
    assert!(bad.is_empty(), "{} disagreements:\n{}", bad.len(), bad.join("\n"));
    assert_eq!(out, expected, "every byte of the output buffer");
    eprintln!("{} window contract checks agree", WHAT.len() + 3);
}

const DISPLAY_WHAT: [&str; 11] = [
    "a window context",
    "a wgpu context",
    "open",
    "a buffer",
    "upload",
    "present",
    "present again",
    "and again",
    "poll",
    "a second open refused",
    "an open with a shader that is not UTF-8 refused",
];

/// With a display: a window opens and a buffer is presented three times.
/// Every answer agrees with the interpreter's; and when `XVFB_FB` names the X
/// server's framebuffer file (`Xvfb -fbdir <dir> -nocursor`: the pointer is
/// drawn into the framebuffer, over wherever the window is placed), the
/// window is found on the screen and every pixel of it is the buffer's. The body leaves the window
/// open so it is still there to read.
#[test]
#[ignore = "needs a display: run with DISPLAY set"]
fn interpreter_and_runtime_agree_with_a_display() {
    let dir = match std::env::var("BASE_HPROG_DIR") {
        Ok(d) => PathBuf::from(d),
        Err(_) => PathBuf::from(lean_artifacts::DIR),
    };
    let v: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join("hprog_window_display_corpus/expected.json")).expect("read corpus"),
    )
    .expect("corpus shape");
    let expected: Vec<u8> =
        v["expected"].as_array().expect("expected").iter().map(|b| b.as_u64().expect("byte") as u8).collect();
    let (w, h) = (v["width"].as_u64().expect("width") as usize, v["height"].as_u64().expect("height") as usize);
    let a = Artifact::from_bytes(&std::fs::read(dir.join("hprog_window_display_corpus.cbor")).expect("read artifact"))
        .expect("artifact shape");

    let mut b = base::Driver::load(a).expect("compile");
    let mut out = vec![0u8; expected.len()];
    b.execute("main", &[], &mut out).expect("execute");

    let bad: Vec<String> = DISPLAY_WHAT
        .iter()
        .enumerate()
        .filter_map(|(k, what)| {
            let (got, want) = (&out[4 * k..4 * k + 4], &expected[4 * k..4 * k + 4]);
            (got != want).then(|| {
                format!(
                    "  {what}: lean {} != runtime {}",
                    i32::from_le_bytes(want.try_into().unwrap()),
                    i32::from_le_bytes(got.try_into().unwrap())
                )
            })
        })
        .collect();
    assert!(bad.is_empty(), "{} disagreements:\n{}", bad.len(), bad.join("\n"));

    let Ok(fb) = std::env::var("XVFB_FB") else {
        eprintln!("{} window checks agree; no framebuffer named, pixels not read", DISPLAY_WHAT.len());
        return;
    };
    let screen = Screen::read(&PathBuf::from(fb));
    let want = |x: usize, y: usize| [(4 * x) as u8, (5 * y) as u8, 128u8];
    let origin = (0..screen.height)
        .flat_map(|y| (0..screen.width).map(move |x| (x, y)))
        .find(|&(x, y)| {
            x + w <= screen.width
                && y + h <= screen.height
                && screen.rgb(x, y) == want(0, 0)
                && screen.rgb(x + 1, y) == want(1, 0)
                && screen.rgb(x, y + 1) == want(0, 1)
        })
        .expect("the window's first pixels are nowhere on the screen");
    let wrong: Vec<String> = (0..h)
        .flat_map(|y| (0..w).map(move |x| (x, y)))
        .filter(|&(x, y)| screen.rgb(origin.0 + x, origin.1 + y) != want(x, y))
        .map(|(x, y)| format!("({x}, {y}): {:?} not {:?}", screen.rgb(origin.0 + x, origin.1 + y), want(x, y)))
        .collect();
    assert!(
        wrong.is_empty(),
        "{} of {} pixels of the window differ from the buffer, first: {:?}",
        wrong.len(),
        w * h,
        &wrong[..wrong.len().min(8)]
    );
    eprintln!("{} window checks agree; all {} pixels on the screen are the buffer's", DISPLAY_WHAT.len(), w * h);
}

/// An X server's screen, as `Xvfb -fbdir` keeps it: an XWD image of 32-bit
/// pixels.
struct Screen {
    bytes: Vec<u8>,
    data: usize,
    line: usize,
    width: usize,
    height: usize,
}

impl Screen {
    fn read(path: &std::path::Path) -> Screen {
        let bytes = std::fs::read(path).expect("read the framebuffer");
        let field = |k: usize| u32::from_be_bytes(bytes[4 * k..4 * k + 4].try_into().unwrap()) as usize;
        assert_eq!(field(11), 32, "32-bit pixels");
        let (header, colors) = (field(0), field(19));
        Screen { data: header + 12 * colors, line: field(12), width: field(4), height: field(5), bytes }
    }

    /// Red, green and blue at `(x, y)`: stored blue first.
    fn rgb(&self, x: usize, y: usize) -> [u8; 3] {
        let p = self.data + y * self.line + 4 * x;
        [self.bytes[p + 2], self.bytes[p + 1], self.bytes[p]]
    }
}
