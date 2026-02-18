use base_types::{Action, Kind, State, UnitSpec};
use crate::harness::BenchResult;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::thread;
use std::time::Duration;

// ---------------------------------------------------------------------------
// Network Benchmark: TCP Echo Server
//
// Tests network unit correctness and performance.
// Pattern: Client sends N bytes → server echoes back → client verifies.
// ---------------------------------------------------------------------------

const ADDR_OFF: usize = 0x1000;     // Server address string
const LISTENER_OFF: usize = 0x2000; // Listener handle
const CONN_OFF: usize = 0x2008;     // Connection handle
const FLAG1_OFF: usize = 0x3000;    // Bind completion
const FLAG2_OFF: usize = 0x3008;    // Accept completion
const FLAG3_OFF: usize = 0x3010;    // Recv completion
const FLAG4_OFF: usize = 0x3018;    // Send completion
const DATA_OFF: usize = 0x4000;     // Data buffer

/// Generate test data
fn gen_test_data(n: usize) -> Vec<u8> {
    (0..n).map(|i| (i % 256) as u8).collect()
}

/// Build Base TCP echo server algorithm.
///
/// Server actions:
///   [0] NetConnect (bind listener) → LISTENER_OFF
///   [1] NetAccept → CONN_OFF
///   [2] NetRecv (DATA_OFF, size)
///   [3] NetSend (DATA_OFF, size)
///   [4] AsyncDispatch → Network (action 0), flag FLAG1
///   [5] Wait(FLAG1)
///   [6] AsyncDispatch → Network (action 1), flag FLAG2
///   [7] Wait(FLAG2)
///   [8] AsyncDispatch → Network (action 2), flag FLAG3
///   [9] Wait(FLAG3)
///  [10] AsyncDispatch → Network (action 3), flag FLAG4
///  [11] Wait(FLAG4)
fn build_echo_server(port: u16, data_size: usize) -> base::Algorithm {
    let payload_size = DATA_OFF + data_size;
    let mut payloads = vec![0u8; payload_size];

    // Server address
    let addr = format!("0.0.0.0:{}\0", port);
    payloads[ADDR_OFF..ADDR_OFF + addr.len()].copy_from_slice(addr.as_bytes());

    let actions = vec![
        // [0] NetConnect (bind listener)
        Action {
            kind: Kind::NetConnect,
            src: ADDR_OFF as u32,
            dst: LISTENER_OFF as u32,
            offset: 4096,
            size: 0,
        },
        // [1] NetAccept
        Action {
            kind: Kind::NetAccept,
            src: LISTENER_OFF as u32,
            dst: CONN_OFF as u32,
            offset: 0,
            size: 0,
        },
        // [2] NetRecv
        Action {
            kind: Kind::NetRecv,
            src: CONN_OFF as u32,
            dst: DATA_OFF as u32,
            offset: 0,
            size: data_size as u32,
        },
        // [3] NetSend
        Action {
            kind: Kind::NetSend,
            dst: CONN_OFF as u32,
            src: DATA_OFF as u32,
            offset: 0,
            size: data_size as u32,
        },
        // [4] AsyncDispatch → Network (action 0)
        Action {
            kind: Kind::AsyncDispatch,
            dst: 3,  // network_units (dst: 3 = Network)
            src: 0,
            offset: FLAG1_OFF as u32,
            size: 0,
        },
        // [5] Wait bind
        Action {
            kind: Kind::Wait,
            dst: FLAG1_OFF as u32,
            src: 0,
            offset: 0,
            size: 0,
        },
        // [6] AsyncDispatch → Network (action 1)
        Action {
            kind: Kind::AsyncDispatch,
            dst: 3,  // network_units
            src: 1,
            offset: FLAG2_OFF as u32,
            size: 0,
        },
        // [7] Wait accept
        Action {
            kind: Kind::Wait,
            dst: FLAG2_OFF as u32,
            src: 0,
            offset: 0,
            size: 0,
        },
        // [8] AsyncDispatch → Network (action 2)
        Action {
            kind: Kind::AsyncDispatch,
            dst: 3,  // network_units
            src: 2,
            offset: FLAG3_OFF as u32,
            size: 0,
        },
        // [9] Wait recv
        Action {
            kind: Kind::Wait,
            dst: FLAG3_OFF as u32,
            src: 0,
            offset: 0,
            size: 0,
        },
        // [10] AsyncDispatch → Network (action 3)
        Action {
            kind: Kind::AsyncDispatch,
            dst: 3,  // network_units
            src: 3,
            offset: FLAG4_OFF as u32,
            size: 0,
        },
        // [11] Wait send
        Action {
            kind: Kind::Wait,
            dst: FLAG4_OFF as u32,
            src: 0,
            offset: 0,
            size: 0,
        },
    ];

    let num_actions = actions.len();

    base::Algorithm {
        actions,
        payloads,
        state: State {
            regs_per_unit: 0,
            gpu_size: 0,
            computational_regs: 0,
            file_buffer_size: 0,
            gpu_shader_offsets: vec![],
            cranelift_ir_offsets: vec![],
        },
        units: UnitSpec {
            simd_units: 0,
            gpu_units: 0,
            computational_units: 0,
            file_units: 0,
            network_units: 1,
            memory_units: 0,
            ffi_units: 0,
            hash_table_units: 0,
            lmdb_units: 0,
            cranelift_units: 0,
            backends_bits: 0xFFFF_FFFF,
        },
        simd_assignments: vec![],
        computational_assignments: vec![],
        memory_assignments: vec![],
        file_assignments: vec![],
        network_assignments: vec![0; num_actions],
        ffi_assignments: vec![],
        hash_table_assignments: vec![],
        lmdb_assignments: vec![],
        gpu_assignments: vec![],
        cranelift_assignments: vec![],
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

    // If we didn't get all the data, return what we got
    // The verification will catch this as a mismatch
    response[..total_read].to_vec()
}

pub fn print_network_table(results: &[BenchResult]) {
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
        let tokio_str = match r.rust_ms {
            Some(ms) => format!("{:.1}ms", ms),
            None => "N/A".to_string(),
        };
        let base_str = if r.base_ms.is_nan() {
            "N/A".to_string()
        } else {
            format!("{:.1}ms", r.base_ms)
        };
        let check_str = match r.verified {
            Some(true) => "\u{2713}",
            Some(false) => "\u{2717}",
            None => "\u{2014}",
        };

        println!(
            "{:<name_w$} {:>col_w$} {:>col_w$} {:>6}",
            r.name,
            tokio_str,
            base_str,
            check_str,
            name_w = name_w,
            col_w = col_w
        );
    }
    println!();
}

pub fn run(iterations: usize) -> Vec<BenchResult> {
    let mut results = Vec::new();

    eprintln!("\n=== Network Benchmarks: TCP Echo Server ===");
    eprintln!("  Client sends N bytes → server echoes → client verifies\n");

    let base_port = 9000;
    let mut port_counter = 0u16;

    for (bench_idx, &kb) in [1, 10, 100].iter().enumerate() {
        // Give each benchmark size its own port range (100 ports each)
        // to avoid TIME_WAIT conflicts between different benchmark sizes
        port_counter = (bench_idx as u16) * 100;
        let data_size = kb * 1024;
        eprintln!("  TCP Echo {}KB ...", kb);

        let test_data = gen_test_data(data_size);

        // Raw TCP baseline - use unique port per iteration
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

            // Small delay to ensure port is fully released
            thread::sleep(Duration::from_millis(10));
        }

        let mut sorted = raw_tcp_times.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let raw_tcp_ms = sorted[sorted.len() / 2];

        // Base version - use unique port per iteration
        let mut base_times = Vec::new();
        for _iter in 0..iterations {
            let port = base_port + port_counter;
            port_counter += 1;

            let base_alg = build_echo_server(port, data_size);
            let test_data_clone = test_data.clone();

            thread::spawn(move || {
                let _ = base::execute(base_alg);
            });

            let start = std::time::Instant::now();
            let _response = tcp_client(port, &test_data_clone);
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
            base_times.push(elapsed);

            // Small delay to ensure port is fully released
            thread::sleep(Duration::from_millis(10));
        }

        let mut sorted = base_times.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let base_ms = sorted[sorted.len() / 2];

        // Verification: run once more and check response
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
            let alg = build_echo_server(port, data_size);
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

        results.push(BenchResult {
            name: format!("TCP Echo {}KB", kb),
            python_ms: None,
            rust_ms: Some(raw_tcp_ms),
            base_ms,
            verified: Some(verify_raw && verify_base),
            actions: None,
        });
    }

    results
}
