//! A load is total: whatever artifact arrives, `Driver::load` answers a driver
//! or an error, and never panics or aborts. Each case is a generated artifact
//! with one field of its structure changed.

use base::{Artifact, Driver};
use serde_json::Value;

/// xorshift64*, seeded, so a failing case is reproducible from its number.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// Every place in `v` a mutation can land, as a path of indices and keys.
fn places(v: &Value, path: &mut Vec<Value>, out: &mut Vec<Vec<Value>>) {
    out.push(path.clone());
    match v {
        Value::Array(xs) => {
            for (i, x) in xs.iter().enumerate() {
                path.push(i.into());
                places(x, path, out);
                path.pop();
            }
        }
        Value::Object(m) => {
            for (k, x) in m {
                path.push(k.clone().into());
                places(x, path, out);
                path.pop();
            }
        }
        _ => {}
    }
}

fn at<'a>(v: &'a mut Value, path: &[Value]) -> &'a mut Value {
    path.iter().fold(v, |v, p| match p {
        Value::Number(i) => &mut v[i.as_u64().unwrap() as usize],
        Value::String(k) => &mut v[k.as_str()],
        _ => unreachable!(),
    })
}

const NUMBERS: [u64; 8] = [0, 1, 2, 7, 255, u32::MAX as u64, 1 << 40, u64::MAX];
const WORDS: [&str; 8] = ["i64", "i8x16", "iadd", "load", "notrap", "", "f128", "brif"];

fn mutate(v: &mut Value, rng: &mut Rng) {
    let mut all = Vec::new();
    places(v, &mut Vec::new(), &mut all);
    let place = all[rng.below(all.len())].clone();
    let target = at(v, &place);
    *target = match target.take() {
        Value::Number(_) => NUMBERS[rng.below(NUMBERS.len())].into(),
        Value::String(_) => WORDS[rng.below(WORDS.len())].into(),
        Value::Array(mut xs) if !xs.is_empty() => {
            let i = rng.below(xs.len());
            if rng.next() % 2 == 0 {
                xs.remove(i);
            } else {
                let x = xs[i].clone();
                xs.insert(i, x);
            }
            Value::Array(xs)
        }
        other => other,
    };
}

/// Loads `cases` mutations drawn from `seed`.
fn mutations(seed: u64, cases: u32) {
    let seeds: Vec<(&str, Value)> = lean_artifacts::NAMES
        .iter()
        .filter_map(|&n| {
            let bytes = lean_artifacts::by_name(n)?;
            (bytes.len() < 64 << 10).then(|| (n, serde_json::to_value(Artifact::from_bytes(bytes).unwrap()).unwrap()))
        })
        .collect();
    assert!(!seeds.is_empty());
    let mut rng = Rng(seed);
    for case in 0..cases {
        let (name, seed) = &seeds[rng.below(seeds.len())];
        let mut v = seed.clone();
        mutate(&mut v, &mut rng);
        let Ok(artifact) = serde_json::from_value::<Artifact>(v) else { continue };
        let loaded = std::panic::catch_unwind(|| Driver::load(artifact).map(drop));
        assert!(loaded.is_ok(), "case {case}, a mutation of {name}, panicked while loading");
    }
}

#[test]
fn every_mutated_artifact_loads_or_is_refused() {
    mutations(0x9e37_79b9_7f4a_7c15, 200);
}

/// The same, at length: `cargo test -p base --test base -- --ignored totality`.
#[test]
#[ignore]
fn every_mutated_artifact_loads_or_is_refused_at_length() {
    mutations(0xd1b5_4a32_d192_ed03, 20_000);
}
