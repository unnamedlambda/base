"""
GPU kernel benchmarks: PyTorch (cuBLAS) vs py_base (PTX / cuBLAS via CLIF).

GEMV:    torch.mv(A, x) vs py_base cuBLAS SGEMV — persistent pattern.
         A is uploaded once (like torch.cuda tensor), only x transferred per call.
         Both sides time resident compute only.

RMSNorm: manual RMSNorm vs py_base PTX (256-thread block, warp shuffle reduce)
         y[i] = x[i] * w[i] / sqrt(mean(x^2) + eps), eps=1e-5

Softmax: torch.softmax vs py_base PTX (3-phase: max, exp+sum, normalize)
         Uses ex2.approx for fast approximate exp.

Decoder layer: persistent single-token decoder layer with fixed Qwen-like dims.
               Uses 7 cuBLAS GEMVs plus PTX RMSNorm / SiLU-gate / residual-add.
               Attention is simplified to the seq_len=1 case, so attn(v)=v.

Decode attention: persistent batch-1 decode attention over a resident KV cache.
                  Uses batched cuBLAS GEMMs for QK / PV and PTX softmax over
                  attention scores. No RoPE or GQA yet.

PyTorch times are CUDA-synchronized and use resident GPU tensors.
"""

import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import harness
import py_base

try:
    import torch
    _TORCH_OK = torch.cuda.is_available()
except ImportError:
    _TORCH_OK = False


GEMV_SIZES = [(4096, 4096), (4096, 11008), (11008, 4096)]
NORM_SIZES = [896, 2048, 4096]
SOFTMAX_SIZES = [512, 2048, 32000]
ATTN_SEQS = [128, 512, 2048]
D_MODEL = 896
D_HEAD = 64
N_HEADS = 14
D_FFN = 4864
STACK_DEPTHS = [16, 32]
SOFTMAX_INNER_ITERS = 64
ATTN_INNER_ITERS = 64


def _time_torch(fn):
    torch.cuda.synchronize()

    def synced():
        fn()
        torch.cuda.synchronize()

    return harness.time_ms(synced)


def _run_gemv(persist_path: str, rounds: int) -> list[harness.BenchResult]:
    raw = json.loads(open(persist_path).read())
    config = py_base.BaseConfig(json.dumps(raw[0]))
    load_alg = py_base.Algorithm(json.dumps(raw[1]))
    prep_alg = py_base.Algorithm(json.dumps(raw[2]))
    infer_alg = py_base.Algorithm(json.dumps(raw[3]))

    results = []
    rng = np.random.default_rng(42)

    for m, n in GEMV_SIZES:
        engine = py_base.Base(config)
        a_np = rng.standard_normal((m, n)).astype(np.float32)
        x_np = rng.standard_normal(n).astype(np.float32)
        out = bytearray(m * 4)

        load_data = struct.pack("<QQ", m, n) + a_np.tobytes()
        engine.execute(load_alg, load_data)
        x_bytes = x_np.tobytes()

        if _TORCH_OK:
            a_t = torch.from_numpy(a_np).cuda()
            x_t = torch.from_numpy(x_np).cuda()
            torch.mv(a_t, x_t)
            torch_ms = harness.median_of(
                rounds,
                lambda: _time_torch(lambda: torch.mv(a_t, x_t)),
            )
        else:
            torch_ms = None

        engine.execute(prep_alg, x_bytes)
        engine.execute_into(infer_alg, b"", out)
        pybase_ms = harness.median_of(
            rounds,
            lambda: (
                engine.execute(prep_alg, x_bytes),
                harness.time_ms(lambda: engine.execute(infer_alg)),
            )[1],
        )

        if _TORCH_OK:
            engine.execute(prep_alg, x_bytes)
            engine.execute_into(infer_alg, b"", out)
            ref = torch.mv(a_t, x_t).cpu().numpy()
            got = np.frombuffer(bytes(out), dtype=np.float32)
            mag = max(float(np.abs(ref).max()), 1e-6)
            verified = bool(np.max(np.abs(got - ref)) / mag < 1e-2)
        else:
            verified = None

        results.append(
            harness.BenchResult(
                name=f"GEMV    ({m}×{n})",
                python_ms=torch_ms,
                pybase_ms=pybase_ms,
                verified=verified,
            )
        )

    return results


def _run_rmsnorm(algo_path: str, rounds: int) -> list[harness.BenchResult]:
    raw = json.loads(open(algo_path).read())
    config = py_base.BaseConfig(json.dumps(raw[0]))
    load_alg = py_base.Algorithm(json.dumps(raw[1]))
    prep_alg = py_base.Algorithm(json.dumps(raw[2]))
    infer_alg = py_base.Algorithm(json.dumps(raw[3]))
    results = []
    rng = np.random.default_rng(7)

    for n in NORM_SIZES:
        engine = py_base.Base(config)
        x_np = rng.standard_normal(n).astype(np.float32)
        w_np = rng.standard_normal(n).astype(np.float32) * 0.5 + 1.0
        out = bytearray(n * 4)
        load_data = struct.pack("<Q", n) + w_np.tobytes()
        engine.execute(load_alg, load_data)
        x_bytes = x_np.tobytes()

        if _TORCH_OK:
            x_t = torch.from_numpy(x_np).cuda()
            w_t = torch.from_numpy(w_np).cuda()

            def rms(x, w):
                return x * w * torch.rsqrt(x.pow(2).mean() + 1e-5)

            rms(x_t, w_t)
            torch_ms = harness.median_of(
                rounds,
                lambda: _time_torch(lambda: rms(x_t, w_t)),
            )
        else:
            torch_ms = None

        engine.execute(prep_alg, x_bytes)
        engine.execute(infer_alg)
        pybase_ms = harness.median_of(
            rounds,
            lambda: (
                engine.execute(prep_alg, x_bytes),
                harness.time_ms(lambda: engine.execute(infer_alg)),
            )[1],
        )

        if _TORCH_OK:
            engine.execute(prep_alg, x_bytes)
            engine.execute_into(infer_alg, b"", out)
            ref = rms(x_t, w_t).cpu().numpy()
            got = np.frombuffer(bytes(out), dtype=np.float32)
            mag = max(float(np.abs(ref).max()), 1e-6)
            verified = bool(np.max(np.abs(got - ref)) / mag < 1e-2)
        else:
            verified = None

        results.append(
            harness.BenchResult(
                name=f"RMSNorm ({n})",
                python_ms=torch_ms,
                pybase_ms=pybase_ms,
                verified=verified,
            )
        )
    return results


def _run_softmax(algo_path: str, rounds: int) -> list[harness.BenchResult]:
    raw = json.loads(open(algo_path).read())
    config = py_base.BaseConfig(json.dumps(raw[0]))
    load_alg = py_base.Algorithm(json.dumps(raw[1]))
    prep_alg = py_base.Algorithm(json.dumps(raw[2]))
    infer_alg = py_base.Algorithm(json.dumps(raw[3]))
    stack_alg = py_base.Algorithm(json.dumps(raw[4]))
    results = []
    rng = np.random.default_rng(13)

    for n in SOFTMAX_SIZES:
        engine = py_base.Base(config)
        x_np = rng.standard_normal(n).astype(np.float32)
        out = bytearray(n * 4)
        load_data = struct.pack("<Q", n)
        engine.execute(load_alg, load_data)
        x_bytes = x_np.tobytes()

        if _TORCH_OK:
            x_t = torch.from_numpy(x_np).cuda()
            torch.softmax(x_t, dim=0)

            def softmax_many():
                for _ in range(SOFTMAX_INNER_ITERS):
                    torch.softmax(x_t, dim=0)

            torch_ms = harness.median_of(rounds, lambda: _time_torch(softmax_many)) / SOFTMAX_INNER_ITERS
        else:
            torch_ms = None

        engine.execute(prep_alg, x_bytes)
        engine.execute(stack_alg)
        pybase_ms = harness.median_of(
            rounds,
            lambda: (
                engine.execute(prep_alg, x_bytes),
                harness.time_ms(lambda: engine.execute(stack_alg)),
            )[1],
        ) / SOFTMAX_INNER_ITERS

        if _TORCH_OK:
            engine.execute(prep_alg, x_bytes)
            engine.execute_into(stack_alg, b"", out)
            ref = torch.softmax(x_t, dim=0).cpu().numpy()
            got = np.frombuffer(bytes(out), dtype=np.float32)
            verified = bool(np.max(np.abs(got - ref)) < 1e-3)
        else:
            verified = None

        results.append(
            harness.BenchResult(
                name=f"Softmax ({n})",
                python_ms=torch_ms,
                pybase_ms=pybase_ms,
                verified=verified,
            )
        )
    return results


def _rms_torch(x, w):
    return x * w * torch.rsqrt(x.pow(2).mean() + 1e-5)


def _run_decoder_layer(algo_path: str, rounds: int) -> list[harness.BenchResult]:
    raw = json.loads(open(algo_path).read())
    config = py_base.BaseConfig(json.dumps(raw[0]))
    load_alg = py_base.Algorithm(json.dumps(raw[1]))
    prep_alg = py_base.Algorithm(json.dumps(raw[2]))
    infer_alg = py_base.Algorithm(json.dumps(raw[3]))
    stack16_alg = py_base.Algorithm(json.dumps(raw[4]))
    stack32_alg = py_base.Algorithm(json.dumps(raw[5]))
    engine = py_base.Base(config)

    rng = np.random.default_rng(23)

    rms1 = (rng.standard_normal(D_MODEL).astype(np.float32) * 0.5 + 1.0)
    wq = rng.standard_normal((D_MODEL, D_MODEL)).astype(np.float32) * 0.02
    wk = rng.standard_normal((D_MODEL, D_MODEL)).astype(np.float32) * 0.02
    wv = rng.standard_normal((D_MODEL, D_MODEL)).astype(np.float32) * 0.02
    wo = rng.standard_normal((D_MODEL, D_MODEL)).astype(np.float32) * 0.02
    rms2 = (rng.standard_normal(D_MODEL).astype(np.float32) * 0.5 + 1.0)
    wg = rng.standard_normal((D_FFN, D_MODEL)).astype(np.float32) * 0.02
    wu = rng.standard_normal((D_FFN, D_MODEL)).astype(np.float32) * 0.02
    wd = rng.standard_normal((D_MODEL, D_FFN)).astype(np.float32) * 0.02

    x_np = rng.standard_normal(D_MODEL).astype(np.float32)

    load_data = b"".join(
        [
            rms1.tobytes(),
            wq.tobytes(),
            wk.tobytes(),
            wv.tobytes(),
            wo.tobytes(),
            rms2.tobytes(),
            wg.tobytes(),
            wu.tobytes(),
            wd.tobytes(),
        ]
    )
    engine.execute(load_alg, load_data)
    x_bytes = x_np.tobytes()
    out = bytearray(D_MODEL * 4)

    if _TORCH_OK:
        x_t = torch.from_numpy(x_np).cuda()
        rms1_t = torch.from_numpy(rms1).cuda()
        wq_t = torch.from_numpy(wq).cuda()
        wk_t = torch.from_numpy(wk).cuda()
        wv_t = torch.from_numpy(wv).cuda()
        wo_t = torch.from_numpy(wo).cuda()
        rms2_t = torch.from_numpy(rms2).cuda()
        wg_t = torch.from_numpy(wg).cuda()
        wu_t = torch.from_numpy(wu).cuda()
        wd_t = torch.from_numpy(wd).cuda()

        def layer():
            x1 = _rms_torch(x_t, rms1_t)
            _q = torch.mv(wq_t, x1)
            _k = torch.mv(wk_t, x1)
            v = torch.mv(wv_t, x1)
            o = torch.mv(wo_t, v)
            h = x_t + o
            x2 = _rms_torch(h, rms2_t)
            g = torch.mv(wg_t, x2)
            u = torch.mv(wu_t, x2)
            a = torch.nn.functional.silu(g) * u
            d = torch.mv(wd_t, a)
            return h + d

        layer()
        torch_ms = harness.median_of(rounds, lambda: _time_torch(layer))
    else:
        torch_ms = None

    engine.execute(prep_alg, x_bytes)
    engine.execute(infer_alg)
    pybase_ms = harness.median_of(
        rounds,
        lambda: (
            engine.execute(prep_alg, x_bytes),
            harness.time_ms(lambda: engine.execute(infer_alg)),
        )[1],
    )

    if _TORCH_OK:
        engine.execute(prep_alg, x_bytes)
        engine.execute_into(infer_alg, b"", out)
        ref = layer().cpu().numpy()
        got = np.frombuffer(bytes(out), dtype=np.float32)
        mag = max(float(np.abs(ref).max()), 1e-6)
        verified = bool(np.max(np.abs(got - ref)) / mag < 2e-2)
    else:
        verified = None

    results = [
        harness.BenchResult(
            name=f"Layer   ({D_MODEL}->{D_FFN}->{D_MODEL})",
            python_ms=torch_ms,
            pybase_ms=pybase_ms,
            verified=verified,
        )
    ]

    if _TORCH_OK:

        def stack(depth: int):
            h = x_t
            for _ in range(depth):
                x1 = _rms_torch(h, rms1_t)
                _q = torch.mv(wq_t, x1)
                _k = torch.mv(wk_t, x1)
                v = torch.mv(wv_t, x1)
                o = torch.mv(wo_t, v)
                h = h + o
                x2 = _rms_torch(h, rms2_t)
                g = torch.mv(wg_t, x2)
                u = torch.mv(wu_t, x2)
                a = torch.nn.functional.silu(g) * u
                d = torch.mv(wd_t, a)
                h = h + d
            return h

    for depth in STACK_DEPTHS:
        stack_alg = stack16_alg if depth == 16 else stack32_alg

        if _TORCH_OK:
            stack(depth)
            torch_stack_ms = harness.median_of(rounds, lambda: _time_torch(lambda: stack(depth)))
        else:
            torch_stack_ms = None

        engine.execute(prep_alg, x_bytes)
        engine.execute(stack_alg)
        pybase_stack_ms = harness.median_of(
            rounds,
            lambda: (
                engine.execute(prep_alg, x_bytes),
                harness.time_ms(lambda: engine.execute(stack_alg)),
            )[1],
        )

        if _TORCH_OK:
            engine.execute(prep_alg, x_bytes)
            engine.execute_into(stack_alg, b"", out)
            ref = stack(depth).cpu().numpy()
            got = np.frombuffer(bytes(out), dtype=np.float32)
            mag = max(float(np.abs(ref).max()), 1e-6)
            stack_ok = bool(np.max(np.abs(got - ref)) / mag < 2e-2)
        else:
            stack_ok = None

        results.append(
            harness.BenchResult(
                name=f"Stack{depth} ({D_MODEL})",
                python_ms=torch_stack_ms,
                pybase_ms=pybase_stack_ms,
                verified=stack_ok,
            )
        )

    return results


def _run_decode_attention(algo_path: str, rounds: int) -> list[harness.BenchResult]:
    raw = json.loads(open(algo_path).read())
    config = py_base.BaseConfig(json.dumps(raw[0]))
    load_alg = py_base.Algorithm(json.dumps(raw[1]))
    prep_alg = py_base.Algorithm(json.dumps(raw[2]))
    infer_alg = py_base.Algorithm(json.dumps(raw[3]))
    stack_alg = py_base.Algorithm(json.dumps(raw[4]))
    results = []
    rng = np.random.default_rng(29)

    scale = float(D_HEAD ** -0.5)

    for seq_len in ATTN_SEQS:
        engine = py_base.Base(config)
        q_np = rng.standard_normal((N_HEADS, D_HEAD)).astype(np.float32) * 0.2
        k_np = rng.standard_normal((N_HEADS, seq_len, D_HEAD)).astype(np.float32) * 0.2
        v_np = rng.standard_normal((N_HEADS, seq_len, D_HEAD)).astype(np.float32) * 0.2
        out = bytearray(D_MODEL * 4)

        load_data = struct.pack("<Q", seq_len) + k_np.tobytes() + v_np.tobytes()
        engine.execute(load_alg, load_data)
        q_bytes = q_np.reshape(-1).tobytes()

        if _TORCH_OK:
            q_t = torch.from_numpy(q_np).cuda()
            k_t = torch.from_numpy(k_np).cuda()
            v_t = torch.from_numpy(v_np).cuda()

            def decode_attn():
                scores = torch.bmm(k_t, q_t.unsqueeze(2)).squeeze(2) * scale
                probs = torch.softmax(scores, dim=1)
                return torch.bmm(probs.unsqueeze(1), v_t).squeeze(1).reshape(-1)

            decode_attn()

            def decode_attn_many():
                for _ in range(ATTN_INNER_ITERS):
                    decode_attn()

            torch_ms = harness.median_of(rounds, lambda: _time_torch(decode_attn_many)) / ATTN_INNER_ITERS
        else:
            torch_ms = None

        engine.execute(prep_alg, q_bytes)
        engine.execute(stack_alg)
        pybase_ms = harness.median_of(
            rounds,
            lambda: (
                engine.execute(prep_alg, q_bytes),
                harness.time_ms(lambda: engine.execute(stack_alg)),
            )[1],
        ) / ATTN_INNER_ITERS

        if _TORCH_OK:
            engine.execute(prep_alg, q_bytes)
            engine.execute_into(stack_alg, b"", out)
            ref = decode_attn().cpu().numpy()
            got = np.frombuffer(bytes(out), dtype=np.float32)
            mag = max(float(np.abs(ref).max()), 1e-6)
            verified = bool(np.max(np.abs(got - ref)) / mag < 2e-2)
        else:
            verified = None

        results.append(
            harness.BenchResult(
                name=f"Decode  (ctx={seq_len})",
                python_ms=torch_ms,
                pybase_ms=pybase_ms,
                verified=verified,
            )
        )

    return results


def run(
    gemv_persist_path: str,
    rmsnorm_path: str,
    softmax_path: str,
    decoder_layer_path: str,
    decode_attention_path: str,
    rounds: int,
) -> list[harness.BenchResult]:
    return (
        _run_gemv(gemv_persist_path, rounds)
        + _run_rmsnorm(rmsnorm_path, rounds)
        + _run_softmax(softmax_path, rounds)
        + _run_decoder_layer(decoder_layer_path, rounds)
        + _run_decode_attention(decode_attention_path, rounds)
    )
