// Two owner crates, no build script: each is a plain dependency.
fn main() {
    println!("{}", alpha_artifacts::ALPHA.len() + beta_artifacts::BETA.len());
}
