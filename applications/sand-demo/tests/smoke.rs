use std::time::Duration;

#[test]
fn app_launches_and_runs() {
    if std::env::var_os("DISPLAY").is_none() && std::env::var_os("WAYLAND_DISPLAY").is_none() {
        eprintln!("no display; skipping app_launches_and_runs");
        return;
    }
    let mut child = crate::app("sand-demo").spawn().expect("spawn");
    std::thread::sleep(Duration::from_millis(1500));
    match child.try_wait().expect("try_wait") {
        Some(s) => panic!("sand-demo exited early ({s})"),
        None => { let _ = child.kill(); let _ = child.wait(); }
    }
}
