//! The serial library, called directly, checked against a terminal.
//!
//! `lean/algorithms/Host/SerialCorpus.lean` opens the port whose path it is
//! handed, reads what is waiting, writes `ping` and takes the refusals,
//! storing every answer. Here the port is a pseudo-terminal: its far end
//! sends the corpus's greeting before the program runs, and what reaches it
//! afterwards must be exactly what the model says was sent.

use base_types::Artifact;
use std::path::PathBuf;

fn corpus_dir() -> PathBuf {
    match std::env::var("BASE_HPROG_DIR") {
        Ok(d) => PathBuf::from(d),
        Err(_) => PathBuf::from(lean_artifacts::DIR),
    }
}

/// A pseudo-terminal: its far end, and the path and a descriptor of the
/// near end, which is kept open in raw mode so what the far end sends waits
/// unaltered.
#[cfg(target_os = "linux")]
fn terminal() -> (libc::c_int, String, libc::c_int) {
    unsafe {
        let far = libc::posix_openpt(libc::O_RDWR | libc::O_NOCTTY);
        assert!(far >= 0, "posix_openpt");
        assert_eq!(libc::grantpt(far), 0, "grantpt");
        assert_eq!(libc::unlockpt(far), 0, "unlockpt");
        let mut name = [0 as libc::c_char; 128];
        assert_eq!(libc::ptsname_r(far, name.as_mut_ptr(), name.len()), 0, "ptsname_r");
        let path = std::ffi::CStr::from_ptr(name.as_ptr()).to_string_lossy().into_owned();
        let near = libc::open(name.as_ptr(), libc::O_RDWR | libc::O_NOCTTY);
        assert!(near >= 0, "open the near end");
        let mut t: libc::termios = std::mem::zeroed();
        assert_eq!(libc::tcgetattr(near, &mut t), 0);
        libc::cfmakeraw(&mut t);
        assert_eq!(libc::tcsetattr(near, libc::TCSANOW, &t), 0);
        (far, path, near)
    }
}

#[cfg(target_os = "linux")]
#[test]
fn interpreter_and_terminal_agree() {
    let dir = corpus_dir();
    let v: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join("hprog_serial_corpus/expected.json")).expect("read corpus"),
    )
    .expect("corpus shape");
    let expected: Vec<u8> =
        v["expected"].as_array().expect("expected").iter().map(|b| b.as_u64().expect("byte") as u8).collect();
    let names: Vec<String> =
        v["names"].as_array().expect("names").iter().map(|n| n.as_str().expect("name").to_string()).collect();
    let greeting = v["greeting"].as_str().expect("greeting").as_bytes().to_vec();
    let sent: Vec<u8> = v["sent"].as_array().expect("sent").iter().map(|b| b.as_u64().expect("byte") as u8).collect();
    let a = Artifact::from_bytes(&std::fs::read(dir.join("hprog_serial_corpus.cbor")).expect("read artifact"))
        .expect("artifact shape");

    let (far, path, near) = terminal();
    unsafe {
        assert_eq!(libc::write(far, greeting.as_ptr().cast(), greeting.len()), greeting.len() as isize);
        // The terminal hands what arrives to the near end asynchronously:
        // wait until all of it is there.
        let mut waiting: libc::c_int = 0;
        for _ in 0..500 {
            libc::ioctl(near, libc::FIONREAD, &mut waiting);
            if waiting as usize == greeting.len() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        assert_eq!(waiting as usize, greeting.len(), "the greeting reached the near end");
    }

    let mut input = path.clone().into_bytes();
    input.push(0);
    let mut b = base::Driver::load(a).expect("compile");
    let mut out = vec![0u8; expected.len()];
    b.execute("main", &input, &mut out).expect("execute");

    let word = |bs: &[u8], k: usize| i32::from_le_bytes(bs[4 * k..4 * k + 4].try_into().unwrap());
    let bad: Vec<String> = names
        .iter()
        .enumerate()
        .filter(|&(k, _)| word(&out, k) != word(&expected, k))
        .map(|(k, name)| format!("  {name}: lean {} != terminal {}", word(&expected, k), word(&out, k)))
        .collect();
    assert!(bad.is_empty(), "{} results disagree:\n{}", bad.len(), bad.join("\n"));

    let mut got = vec![0u8; 64];
    let n = unsafe {
        libc::fcntl(far, libc::F_SETFL, libc::O_NONBLOCK);
        libc::read(far, got.as_mut_ptr().cast(), got.len())
    };
    got.truncate(n.max(0) as usize);
    assert_eq!(got, sent, "what reached the terminal");
    unsafe {
        libc::close(near);
        libc::close(far);
    }
    eprintln!("{} serial answers agree, and the terminal received {:?}", names.len(), String::from_utf8_lossy(&got));
}
