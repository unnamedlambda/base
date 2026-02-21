use base_types::{Action, Kind, State, UnitSpec};
use std::io::{Read, Write};
use std::net::TcpStream;
use std::thread;
use std::time::Duration;

// ---------------------------------------------------------------------------
// Network Benchmark: TCP Echo Server
//
// Tests network performance via CLIF JIT + extern "C" network primitives.
// Pattern: Client sends N bytes → server echoes back → client verifies.
// ---------------------------------------------------------------------------

// CLIF memory layout (context pointer at offset 0, managed by cl_net_init/cleanup)
const CLIF_DSIZE_OFF: usize = 0x08;   // i64: data buffer size in bytes
const CLIF_ADDR_OFF: usize = 0x1000;  // Server address string (null-terminated)
const CLIF_FLAG_OFF: usize = 0x1F00;  // Cranelift completion flag
const CLIF_IR_OFF: usize = 0x2000;    // CLIF IR source (null-terminated)
const CLIF_DATA_OFF: usize = 0x4000;  // Data buffer

pub struct NetworkBenchResult {
    pub name: String,
    pub raw_tcp_ms: f64,
    pub base_ms: f64,
    pub verified: bool,
}

/// Generate test data
fn gen_test_data(n: usize) -> Vec<u8> {
    (0..n).map(|i| (i % 256) as u8).collect()
}

// ---------------------------------------------------------------------------
// CLIF+Net: Cranelift JIT calls extern "C" network primitives via execute().
// Single compiled function does:
//   init → listen → accept → recv → send → cleanup
// ---------------------------------------------------------------------------

/// Generate CLIF IR for the TCP echo server.
fn gen_net_clif_ir() -> String {
    format!(
r#"function u0:0(i64) system_v {{
    sig0 = (i64) system_v
    sig1 = (i64, i64) -> i64 system_v
    sig2 = (i64, i64) -> i64 system_v
    sig3 = (i64, i64, i64, i64) -> i64 system_v

    fn0 = %cl_net_init sig0
    fn1 = %cl_net_listen sig1
    fn2 = %cl_net_accept sig2
    fn3 = %cl_net_recv sig3
    fn4 = %cl_net_send sig3
    fn5 = %cl_net_cleanup sig0

block0(v0: i64):
    ; --- init network context ---
    call fn0(v0)

    ; --- listen ---
    v1 = iconst.i64 {addr_off}
    v2 = call fn1(v0, v1)

    ; --- accept ---
    v3 = call fn2(v0, v2)

    ; --- recv ---
    v4 = iconst.i64 {data_off}
    v5 = load.i64 v0+{dsize_off}
    v6 = call fn3(v0, v3, v4, v5)

    ; --- send ---
    v7 = call fn4(v0, v3, v4, v5)

    ; --- cleanup ---
    call fn5(v0)
    return
}}"#,
        addr_off = CLIF_ADDR_OFF,
        data_off = CLIF_DATA_OFF,
        dsize_off = CLIF_DSIZE_OFF,
    )
}

/// Build a CLIF+Net algorithm that goes through base::execute().
fn build_clif_net_algorithm(port: u16, data_size: usize) -> base::Algorithm {
    let clif_source = gen_net_clif_ir();
    let clif_bytes = format!("{}\0", clif_source).into_bytes();
    assert!(clif_bytes.len() < (CLIF_DATA_OFF - CLIF_IR_OFF),
        "CLIF IR too large: {} bytes", clif_bytes.len());

    let payload_size = CLIF_DATA_OFF + data_size;
    let mut payloads = vec![0u8; payload_size];

    // CLIF IR source
    payloads[CLIF_IR_OFF..CLIF_IR_OFF + clif_bytes.len()].copy_from_slice(&clif_bytes);

    // Data size parameter
    payloads[CLIF_DSIZE_OFF..CLIF_DSIZE_OFF + 8].copy_from_slice(&(data_size as i64).to_le_bytes());

    // Server address
    let addr = format!("0.0.0.0:{}\0", port);
    payloads[CLIF_ADDR_OFF..CLIF_ADDR_OFF + addr.len()].copy_from_slice(addr.as_bytes());

    let actions = vec![
        Action { kind: Kind::MemCopy, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::AsyncDispatch, dst: 9, src: 0, offset: CLIF_FLAG_OFF as u32, size: 0 },
        Action { kind: Kind::Wait, dst: CLIF_FLAG_OFF as u32, src: 0, offset: 0, size: 0 },
    ];
    let num_actions = actions.len();

    base::Algorithm {
        actions,
        payloads,
        state: State {
            gpu_size: 0,
            file_buffer_size: 0,
            gpu_shader_offsets: vec![],
            cranelift_ir_offsets: vec![CLIF_IR_OFF],
        },
        units: UnitSpec {
            gpu_units: 0,
            file_units: 0,
            memory_units: 0,
            ffi_units: 0,
            cranelift_units: 1,
            backends_bits: 0,
        },
        memory_assignments: vec![],
        file_assignments: vec![],
        ffi_assignments: vec![],
        gpu_assignments: vec![],
        cranelift_assignments: vec![0; num_actions],
        worker_threads: Some(1),
        blocking_threads: Some(1),
        stack_size: Some(256 * 1024),
        timeout_ms: Some(120_000),
        thread_name_prefix: Some("net-bench".into()),
    }
}

/// Raw TCP echo server
fn raw_tcp_echo_server(port: u16) {
    use std::net::{TcpListener, Shutdown};
    let listener = TcpListener::bind(format!("0.0.0.0:{}", port)).unwrap();
    let (mut stream, _) = listener.accept().unwrap();

    let mut buf = vec![0u8; 1024 * 1024];
    let n = stream.read(&mut buf).unwrap();
    stream.write_all(&buf[..n]).unwrap();
    stream.flush().unwrap();

    // Wait for client to finish reading before closing
    stream.shutdown(Shutdown::Write).ok();
    let _ = stream.read(&mut [0u8; 1]);  // Wait for client to close their end
}

/// TCP client: send data and receive echo
fn tcp_client(port: u16, data: &[u8]) -> Vec<u8> {
    use std::net::Shutdown;

    // Retry connecting for up to 2 seconds
    let mut stream = None;
    for attempt in 0..40 {
        match TcpStream::connect(format!("127.0.0.1:{}", port)) {
            Ok(s) => {
                stream = Some(s);
                break;
            }
            Err(e) => {
                if attempt == 39 {
                    eprintln!("Failed to connect to port {}: {}", port, e);
                }
                thread::sleep(Duration::from_millis(50))
            },
        }
    }
    let mut stream = stream.expect("Failed to connect to server");

    stream.write_all(data).unwrap();
    stream.flush().unwrap();
    stream.shutdown(Shutdown::Write).ok();

    // Read response - may come in multiple chunks
    let mut response = vec![0u8; data.len()];
    let mut total_read = 0;
    while total_read < data.len() {
        match stream.read(&mut response[total_read..]) {
            Ok(0) => break,  // EOF
            Ok(n) => total_read += n,
            Err(e) => {
                eprintln!("Read error on port {}: {} (got {} of {} bytes)", port, e, total_read, data.len());
                break;
            }
        }
    }

    stream.shutdown(Shutdown::Both).ok();

    response[..total_read].to_vec()
}

pub fn print_network_table(results: &[NetworkBenchResult]) {
    let name_w = 28;
    let col_w = 14;

    println!();
    println!(
        "{:<name_w$} {:>col_w$} {:>col_w$} {:>6}",
        "Network Benchmark",
        "Raw TCP",
        "Base",
        "Check",
        name_w = name_w,
        col_w = col_w
    );
    println!("{}", "-".repeat(name_w + col_w * 2 + 6 + 3));

    for r in results {
        let check_str = if r.verified { "\u{2713}" } else { "\u{2717}" };

        println!(
            "{:<name_w$} {:>col_w$} {:>col_w$} {:>6}",
            r.name,
            format!("{:.1}ms", r.raw_tcp_ms),
            format!("{:.1}ms", r.base_ms),
            check_str,
            name_w = name_w,
            col_w = col_w
        );
    }
    println!();
}

pub fn run(iterations: usize) -> Vec<NetworkBenchResult> {
    let mut results = Vec::new();

    eprintln!("\n=== Network Benchmarks: TCP Echo Server ===");
    eprintln!("  Client sends N bytes → server echoes → client verifies");
    eprintln!("  Base: Cranelift JIT calls extern C network primitives\n");

    let base_port = 9000;
    let mut port_counter = 0u16;

    for (bench_idx, &kb) in [1, 10, 100].iter().enumerate() {
        // Give each benchmark size its own port range (200 ports each)
        // to avoid TIME_WAIT conflicts between different benchmark sizes
        port_counter = (bench_idx as u16) * 200;
        let data_size = kb * 1024;
        eprintln!("  TCP Echo {}KB ...", kb);

        let test_data = gen_test_data(data_size);

        // --- Raw TCP baseline ---
        let mut raw_tcp_times = Vec::new();
        for _iter in 0..iterations {
            let port = base_port + port_counter;
            port_counter += 1;

            let test_data_clone = test_data.clone();
            let handle = thread::spawn(move || {
                raw_tcp_echo_server(port);
            });

            let start = std::time::Instant::now();
            let _response = tcp_client(port, &test_data_clone);
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;

            handle.join().ok();
            raw_tcp_times.push(elapsed);
            thread::sleep(Duration::from_millis(10));
        }

        let mut sorted = raw_tcp_times.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let raw_tcp_ms = sorted[sorted.len() / 2];

        // --- Base (CLIF+Net) ---
        let mut base_times = Vec::new();
        for _iter in 0..iterations {
            let port = base_port + port_counter;
            port_counter += 1;

            let alg = build_clif_net_algorithm(port, data_size);
            let test_data_clone = test_data.clone();

            thread::spawn(move || {
                let _ = base::execute(alg);
            });

            let start = std::time::Instant::now();
            let _response = tcp_client(port, &test_data_clone);
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
            base_times.push(elapsed);
            thread::sleep(Duration::from_millis(10));
        }

        let mut sorted = base_times.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let base_ms = sorted[sorted.len() / 2];

        // --- Verification ---
        let verify_raw = {
            let port = base_port + port_counter;
            port_counter += 1;
            let test_data_clone = test_data.clone();
            let handle = thread::spawn(move || {
                raw_tcp_echo_server(port);
            });
            let response = tcp_client(port, &test_data_clone);
            handle.join().ok();
            response == test_data
        };

        let verify_base = {
            let port = base_port + port_counter;
            port_counter += 1;
            let test_data_clone = test_data.clone();
            let alg = build_clif_net_algorithm(port, data_size);
            thread::spawn(move || {
                let _ = base::execute(alg);
            });
            let response = tcp_client(port, &test_data_clone);
            response == test_data
        };

        if !verify_raw {
            eprintln!("  VERIFY FAIL: Raw TCP echo incorrect");
        }
        if !verify_base {
            eprintln!("  VERIFY FAIL: Base echo incorrect");
        }

        results.push(NetworkBenchResult {
            name: format!("TCP Echo {}KB", kb),
            raw_tcp_ms,
            base_ms,
            verified: verify_raw && verify_base,
        });
    }

    results
}
