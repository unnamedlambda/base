//! Compiles `src/shim.c` against zimg and libvips, found by pkg-config.

use std::process::Command;

fn pkg(args: &[&str]) -> Vec<String> {
    let out = Command::new("pkg-config").args(args).args(["zimg", "vips"]).output().expect("pkg-config runs");
    assert!(out.status.success(), "zimg and libvips are installed");
    String::from_utf8(out.stdout).unwrap().split_whitespace().map(str::to_owned).collect()
}

fn main() {
    println!("cargo:rerun-if-changed=src/shim.c");
    let mut b = cc::Build::new();
    b.file("src/shim.c").opt_level(3);
    for f in pkg(&["--cflags"]) {
        b.flag(&f);
    }
    b.compile("shim");
    for f in pkg(&["--libs"]) {
        if let Some(l) = f.strip_prefix("-l") {
            println!("cargo:rustc-link-lib={l}");
        }
    }
}
