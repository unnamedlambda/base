//! The artifacts this Lake package generates, a `pub static` each named in
//! capitals; [`by_name`] finds one at run time, and [`DIR`] is where they are.

include!(concat!(env!("OUT_DIR"), "/artifacts.rs"));
