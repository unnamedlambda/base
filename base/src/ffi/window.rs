//! Windows and their input, through winit: the C functions a program's
//! `Ext.window` calls reach, by symbol.
//!
//! winit is a Rust API, so this is the one library here written for the
//! engine: seven functions over it, each a plain C call that the program's
//! `Lib.Window` makes and `Host/WindowLib.lean` states. Everything a window
//! shows — the surface, made from the window's handle, the blit, presenting —
//! is the program's own wgpu calls, and turning these records into the
//! engine's events is the program's too.
//!
//! A window is the program's: `base_window_open` hands back a box it passes
//! to every other call, and `base_window_close` drops. What is kept here is
//! what winit keeps per process: the event loop, made on the first thread
//! that asks and living as long as the thread, and each open window's events
//! not yet read. winit makes one event loop per process, so windows live on
//! that thread; on any other `base_window_init` answers `false`.
//!
//! **Records.** Each event is four little-endian `i64`s: a kind, then its
//! operands.
//!
//! | kind | event | operands |
//! |---|---|---|
//! | 1 | close requested | |
//! | 2 | size in pixels changed | width, height |
//! | 3 / 4 | key pressed / released | the key's USB HID usage (page 7) |
//! | 5 | pointer moved | x, y in pixels, as `f64` bits |
//! | 6 / 7 | button pressed / released | 1 left, 2 right, 3 middle |
//!
//! A key without a HID usage here, and any other button or event, is dropped.

use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::ffi::c_char;
use std::sync::Arc;
use std::time::Duration;

use winit::dpi::LogicalSize;
use winit::event::{ElementState, Event, MouseButton, WindowEvent};
use winit::event_loop::EventLoop;
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::platform::pump_events::EventLoopExtPumpEvents;
use winit::window::{Window, WindowId};

type Record = [i64; 4];

struct Loop {
    events: EventLoop<()>,
    pending: HashMap<WindowId, VecDeque<Record>>,
}

thread_local! {
    static LOOP: RefCell<Option<Loop>> = const { RefCell::new(None) };
}

/// An open window, as the program holds it. A wgpu surface made from it
/// shares it, so the window lives as long as either.
pub(crate) struct Win {
    window: Arc<Window>,
}

/// The window `win` holds, shared; `None` for no handle.
pub(crate) unsafe fn shared(win: i64) -> Option<Arc<Window>> {
    (win as *const Win).as_ref().map(|w| w.window.clone())
}

fn new_event_loop() -> Option<EventLoop<()>> {
    let mut builder = EventLoop::builder();
    #[cfg(all(unix, not(target_vendor = "apple"), not(target_os = "android")))]
    {
        use winit::platform::wayland::EventLoopBuilderExtWayland;
        use winit::platform::x11::EventLoopBuilderExtX11;
        EventLoopBuilderExtX11::with_any_thread(&mut builder, true);
        EventLoopBuilderExtWayland::with_any_thread(&mut builder, true);
    }
    #[cfg(target_os = "windows")]
    {
        use winit::platform::windows::EventLoopBuilderExtWindows;
        builder.with_any_thread(true);
    }
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| builder.build().ok())).ok().flatten()
}

/// The HID usage of a key, `0` for one not here.
fn hid(code: KeyCode) -> i64 {
    use KeyCode as K;
    let letters = [
        K::KeyA, K::KeyB, K::KeyC, K::KeyD, K::KeyE, K::KeyF, K::KeyG, K::KeyH, K::KeyI, K::KeyJ, K::KeyK,
        K::KeyL, K::KeyM, K::KeyN, K::KeyO, K::KeyP, K::KeyQ, K::KeyR, K::KeyS, K::KeyT, K::KeyU, K::KeyV,
        K::KeyW, K::KeyX, K::KeyY, K::KeyZ,
    ];
    let digits =
        [K::Digit1, K::Digit2, K::Digit3, K::Digit4, K::Digit5, K::Digit6, K::Digit7, K::Digit8, K::Digit9, K::Digit0];
    let fkeys = [K::F1, K::F2, K::F3, K::F4, K::F5, K::F6, K::F7, K::F8, K::F9, K::F10, K::F11, K::F12];
    if let Some(i) = letters.iter().position(|&k| k == code) {
        return 4 + i as i64;
    }
    if let Some(i) = digits.iter().position(|&k| k == code) {
        return 30 + i as i64;
    }
    if let Some(i) = fkeys.iter().position(|&k| k == code) {
        return 58 + i as i64;
    }
    match code {
        K::Enter => 40,
        K::Escape => 41,
        K::Backspace => 42,
        K::Tab => 43,
        K::Space => 44,
        K::Minus => 45,
        K::Equal => 46,
        K::BracketLeft => 47,
        K::BracketRight => 48,
        K::Backslash => 49,
        K::Semicolon => 51,
        K::Quote => 52,
        K::Backquote => 53,
        K::Comma => 54,
        K::Period => 55,
        K::Slash => 56,
        K::CapsLock => 57,
        K::Insert => 73,
        K::Home => 74,
        K::PageUp => 75,
        K::Delete => 76,
        K::End => 77,
        K::PageDown => 78,
        K::ArrowRight => 79,
        K::ArrowLeft => 80,
        K::ArrowDown => 81,
        K::ArrowUp => 82,
        K::ControlLeft => 224,
        K::ShiftLeft => 225,
        K::AltLeft => 226,
        K::SuperLeft => 227,
        K::ControlRight => 228,
        K::ShiftRight => 229,
        K::AltRight => 230,
        K::SuperRight => 231,
        _ => 0,
    }
}

fn record(event: WindowEvent) -> Option<Record> {
    let pressed = |s: ElementState| s == ElementState::Pressed;
    match event {
        WindowEvent::CloseRequested => Some([1, 0, 0, 0]),
        WindowEvent::Resized(size) => Some([2, i64::from(size.width), i64::from(size.height), 0]),
        WindowEvent::KeyboardInput { event, .. } => {
            let PhysicalKey::Code(code) = event.physical_key else { return None };
            let usage = hid(code);
            (usage != 0).then(|| [if pressed(event.state) { 3 } else { 4 }, usage, 0, 0])
        }
        WindowEvent::CursorMoved { position, .. } => {
            Some([5, position.x.to_bits() as i64, position.y.to_bits() as i64, 0])
        }
        WindowEvent::MouseInput { state, button, .. } => {
            let b = match button {
                MouseButton::Left => 1,
                MouseButton::Right => 2,
                MouseButton::Middle => 3,
                _ => return None,
            };
            Some([if pressed(state) { 6 } else { 7 }, b, 0, 0])
        }
        _ => None,
    }
}

/// What has arrived, onto each open window's events.
fn pump(l: &mut Loop) {
    let pending = &mut l.pending;
    #[allow(deprecated)]
    let _ = l.events.pump_events(Some(Duration::ZERO), |event, _| {
        if let Event::WindowEvent { window_id, event } = event {
            if let (Some(q), Some(r)) = (pending.get_mut(&window_id), record(event)) {
                q.push_back(r);
            }
        }
    });
}

fn with_loop<T>(f: impl FnOnce(&mut Loop) -> T) -> Option<T> {
    LOOP.with(|l| l.borrow_mut().as_mut().map(f))
}

/// The thread's event loop, started on the first call: `true`, or `false`
/// where there is no display or another thread has it.
unsafe extern "C" fn base_window_init() -> bool {
    LOOP.with(|l| {
        let mut l = l.borrow_mut();
        if l.is_none() {
            *l = new_event_loop().map(|events| Loop { events, pending: HashMap::new() });
        }
        l.is_some()
    })
}

/// A window `width` by `height` logical pixels titled with the C string at
/// `title`, or null — for a size that is not positive too.
unsafe extern "C" fn base_window_open(title: *const c_char, width: i32, height: i32) -> *mut Win {
    if width <= 0 || height <= 0 {
        return std::ptr::null_mut();
    }
    let title = super::read_cstr_ptr(title as *const u8);
    let attrs = Window::default_attributes()
        .with_title(title)
        .with_inner_size(LogicalSize::new(f64::from(width), f64::from(height)));
    with_loop(|l| {
        #[allow(deprecated)]
        let window = l.events.create_window(attrs).ok()?;
        l.pending.insert(window.id(), VecDeque::new());
        let win = Win { window: Arc::new(window) };
        Some(Box::into_raw(Box::new(win)))
    })
    .flatten()
    .unwrap_or(std::ptr::null_mut())
}

/// `win` closed: its window and its events not yet read are gone.
unsafe extern "C" fn base_window_close(win: *mut Win) {
    let win = Box::from_raw(win);
    let id = win.window.id();
    with_loop(|l| l.pending.remove(&id));
    drop(win);
}

/// What has arrived, onto each open window's events.
unsafe extern "C" fn base_window_pump() {
    with_loop(pump);
}

/// What has arrived pumped, and `win`'s next event written at `out` as a
/// record: `true`, or `false` with nothing written when there is none.
unsafe extern "C" fn base_window_poll(win: *mut Win, out: *mut Record) -> bool {
    let id = (*win).window.id();
    let next = with_loop(|l| {
        pump(l);
        l.pending.get_mut(&id).and_then(VecDeque::pop_front)
    })
    .flatten();
    match next {
        Some(r) => {
            std::ptr::write_unaligned(out, r);
            true
        }
        None => false,
    }
}

/// `win`'s size in pixels: the width, and the height in the upper 32 bits.
unsafe extern "C" fn base_window_pixels(win: *mut Win) -> i64 {
    let size = (*win).window.inner_size();
    i64::from(size.width) | (i64::from(size.height) << 32)
}

/// The address of `symbol`, or `None` for one this library does not have.
pub(crate) fn linked(symbol: &str) -> Option<usize> {
    Some(match symbol {
        "base_window_init" => base_window_init as usize,
        "base_window_open" => base_window_open as usize,
        "base_window_close" => base_window_close as usize,
        "base_window_pump" => base_window_pump as usize,
        "base_window_poll" => base_window_poll as usize,
        "base_window_pixels" => base_window_pixels as usize,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hid_usages() {
        assert_eq!(hid(KeyCode::KeyA), 4);
        assert_eq!(hid(KeyCode::KeyZ), 29);
        assert_eq!(hid(KeyCode::Digit1), 30);
        assert_eq!(hid(KeyCode::Digit0), 39);
        assert_eq!(hid(KeyCode::F12), 69);
        assert_eq!(hid(KeyCode::Escape), 41);
        assert_eq!(hid(KeyCode::ArrowUp), 82);
        assert_eq!(hid(KeyCode::NumLock), 0);
    }
}
