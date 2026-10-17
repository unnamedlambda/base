use std::time::Duration;

#[test]
fn app_launches_and_runs() {
    if std::env::var_os("DISPLAY").is_none() && std::env::var_os("WAYLAND_DISPLAY").is_none() {
        eprintln!("no display; skipping app_launches_and_runs");
        return;
    }
    let mut child = crate::app("window-demo")
        .spawn()
        .expect("spawn window-demo");
    std::thread::sleep(Duration::from_millis(1500));
    match child.try_wait().expect("try_wait") {
        Some(status) => panic!("window-demo exited early ({status}) — init/open/present failed"),
        None => {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}
