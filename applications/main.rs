//! Every example application, as one program: `apps <name> [args]`.

#[path = "blackhole/src/main.rs"]
mod blackhole;
#[path = "cli/src/main.rs"]
mod cli;
#[path = "compress/src/main.rs"]
mod compress;
#[path = "draw/src/main.rs"]
mod draw;
#[path = "fft/src/main.rs"]
mod fft;
#[path = "gemv-warp/src/main.rs"]
mod gemv_warp;
#[path = "grad-warp/src/main.rs"]
mod grad_warp;
#[path = "lean4-evaluator/src/main.rs"]
mod lean4_eval;
#[path = "lz4-ptx/src/bin/bench_comp_warp.rs"]
mod lz4_comp_warp;
#[path = "matmul/src/main.rs"]
mod matmul;
#[path = "mlp-cifar/src/main.rs"]
mod mlp_cifar;
#[path = "mlp-warp/src/main.rs"]
mod mlp_warp;
#[path = "raymarch-demo/src/main.rs"]
mod raymarch_demo;
#[path = "raytrace/src/main.rs"]
mod raytrace;
#[path = "sand-demo/src/main.rs"]
mod sand_demo;
#[path = "sat/src/main.rs"]
mod sat;
#[path = "scene/src/main.rs"]
mod scene;
#[path = "sha256/src/main.rs"]
mod sha256;
#[path = "silu-warp/src/main.rs"]
mod silu_warp;
#[path = "warp-demo/src/main.rs"]
mod warp_demo;
#[path = "window-demo/src/main.rs"]
mod window_demo;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let app = if args.is_empty() { String::new() } else { args[0].clone() };
    match app.as_str() {
        "blackhole" => blackhole::main(args),
        "cli" => cli::main(args),
        "compress" => compress::main(args),
        "draw" => draw::main(args),
        "fft" => fft::main(args),
        "gemv-warp" => gemv_warp::main(args),
        "grad-warp" => grad_warp::main(args),
        "lean4-eval" => lean4_eval::main(args),
        "lz4-comp-warp" => lz4_comp_warp::main(args),
        "matmul" => matmul::main(args),
        "mlp-cifar" => mlp_cifar::main(args),
        "mlp-warp" => mlp_warp::main(args),
        "raymarch-demo" => raymarch_demo::main(args),
        "raytrace" => raytrace::main(args),
        "sand-demo" => sand_demo::main(args),
        "sat" => sat::main(args),
        "scene" => scene::main(args),
        "sha256" => sha256::main(args),
        "silu-warp" => silu_warp::main(args),
        "warp-demo" => warp_demo::main(args),
        "window-demo" => window_demo::main(args),
        _ => {
        eprintln!("usage: apps <name> [args]; the names are:");
            eprintln!("  blackhole");
            eprintln!("  cli");
            eprintln!("  compress");
            eprintln!("  draw");
            eprintln!("  fft");
            eprintln!("  gemv-warp");
            eprintln!("  grad-warp");
            eprintln!("  lean4-eval");
            eprintln!("  lz4-comp-warp");
            eprintln!("  matmul");
            eprintln!("  mlp-cifar");
            eprintln!("  mlp-warp");
            eprintln!("  raymarch-demo");
            eprintln!("  raytrace");
            eprintln!("  sand-demo");
            eprintln!("  sat");
            eprintln!("  scene");
            eprintln!("  sha256");
            eprintln!("  silu-warp");
            eprintln!("  warp-demo");
            eprintln!("  window-demo");
            std::process::exit(2);
        }
    }
}
