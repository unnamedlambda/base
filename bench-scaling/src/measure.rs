
use std::path::Path;
use std::process::Command;

pub struct Meas {
    pub secs: Option<f64>,
    pub rss_mb: Option<f64>,
    pub stdout: String,
    pub ok: bool,
    pub timed_out: bool,
}

impl Meas {
    pub fn failed() -> Self {
        Meas { secs: None, rss_mb: None, stdout: String::new(), ok: false, timed_out: false }
    }
}

/// Run `cmd` under a timeout, returning wall seconds and peak RSS.
pub fn measure(
    cmd: &[String],
    cwd: Option<&Path>,
    env: &[(String, String)],
    timeout_s: u64,
) -> Meas {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let tf = std::env::temp_dir().join(format!("scal_{}_{}", std::process::id(), stamp));

    let mut c = Command::new("/usr/bin/time");
    c.arg("-f").arg("%e %M").arg("-o").arg(&tf);
    c.arg("timeout").arg(timeout_s.to_string());
    for a in cmd {
        c.arg(a);
    }
    if let Some(d) = cwd {
        c.current_dir(d);
    }
    for (k, v) in env {
        c.env(k, v);
    }

    let out = match c.output() {
        Ok(o) => o,
        Err(_) => return Meas::failed(),
    };
    let ok = out.status.success();
    // coreutils `timeout` exits 124 when it kills the child.
    let timed_out = out.status.code() == Some(124);
    let mut stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&out.stderr);
    if !stderr.trim().is_empty() {
        stdout.push_str(&stderr);
    }

    let (mut secs, mut rss_mb) = (None, None);
    if let Ok(txt) = std::fs::read_to_string(&tf) {
        // `time` appends its line last; anything before it is the child's stderr.
        if let Some(line) = txt.lines().filter(|l| !l.trim().is_empty()).last() {
            let f: Vec<&str> = line.split_whitespace().collect();
            if f.len() == 2 {
                secs = f[0].parse::<f64>().ok();
                rss_mb = f[1].parse::<f64>().ok().map(|k| k / 1024.0);
            }
        }
        let _ = std::fs::remove_file(&tf);
    }
    Meas { secs, rss_mb, stdout, ok, timed_out }
}

/// Convenience wrapper for the common `&str` call site.
pub fn run(cmd: &[&str], cwd: Option<&Path>, env: &[(&str, &str)], timeout_s: u64) -> Meas {
    let c: Vec<String> = cmd.iter().map(|s| s.to_string()).collect();
    let e: Vec<(String, String)> =
        env.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
    measure(&c, cwd, &e, timeout_s)
}
