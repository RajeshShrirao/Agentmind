"""
test_parallel_scan.py — Verify chunked parallel SSM scan matches sequential scan.

Run: python3 test_parallel_scan.py
"""
import mlx.core as mx
import mlx.nn as nn
import time
import sys
sys.path.insert(0, ".")


def sequential_scan(dA, dBx, h_init=None):
    """Original sequential scan — ground truth."""
    B, L, D, N = dA.shape
    h = h_init if h_init is not None else mx.zeros((B, D, N))
    h_seq = []
    for t in range(L):
        h = dA[:, t] * h + dBx[:, t]
        h_seq.append(h)
    h_stack = mx.stack(h_seq, axis=1)
    return h_stack, h_stack[:, -1]


def chunked_scan(dA, dBx, h_init=None, chunk_size=32):
    """Chunked parallel scan — optimized version."""
    B, L, D, N = dA.shape
    h = h_init if h_init is not None else mx.zeros((B, D, N))
    h_chunks = []

    for t0 in range(0, L, chunk_size):
        t1 = min(t0 + chunk_size, L)
        dA_c = dA[:, t0:t1]
        dBx_c = dBx[:, t0:t1]

        log_dA_c = mx.log(mx.maximum(dA_c, 1e-30))
        cum_log_c = mx.cumsum(log_dA_c, axis=1)
        cum_dA_c = mx.exp(cum_log_c)

        inv_cum_c = mx.exp(-cum_log_c)
        h_chunk = cum_dA_c * mx.cumsum(dBx_c * inv_cum_c, axis=1)
        h_chunk = h_chunk + cum_dA_c * h[:, None, :, :]

        h_chunks.append(h_chunk)
        h = h_chunk[:, -1]

    h_stack = mx.concatenate(h_chunks, axis=1)
    return h_stack, h_stack[:, -1]


def test_parity(name, B, L, D, N, use_h_init=False, atol=1e-3):
    """Run both scans with identical inputs and compare."""
    mx.random.seed(42)
    dA = mx.random.uniform(shape=(B, L, D, N), low=0.5, high=0.999)
    dBx = mx.random.normal(shape=(B, L, D, N)) * 0.1
    h_init = mx.random.normal(shape=(B, D, N)) * 0.01 if use_h_init else None

    h_seq, last_seq = sequential_scan(dA, dBx, h_init)
    h_par, last_par = chunked_scan(dA, dBx, h_init)

    mx.eval(h_seq, h_par, last_seq, last_par)

    max_diff = mx.max(mx.abs(h_seq - h_par)).item()
    last_diff = mx.max(mx.abs(last_seq - last_par)).item()

    passed = max_diff < atol and last_diff < atol
    status = "✅ PASS" if passed else "❌ FAIL"

    print(f"  {name}: max_diff={max_diff:.2e}, last_diff={last_diff:.2e} {status}")
    return passed


def test_speed(B, L, D, N, n_iters=5):
    """Benchmark sequential vs chunked."""
    mx.random.seed(42)
    dA = mx.random.uniform(shape=(B, L, D, N), low=0.5, high=0.999)
    dBx = mx.random.normal(shape=(B, L, D, N)) * 0.1

    # Warmup
    _ = sequential_scan(dA, dBx)
    mx.eval(_[0])
    _ = chunked_scan(dA, dBx)
    mx.eval(_[0])

    # Sequential
    t0 = time.time()
    for _ in range(n_iters):
        h, _ = sequential_scan(dA, dBx)
        mx.eval(h)
    seq_time = (time.time() - t0) / n_iters

    # Chunked
    t0 = time.time()
    for _ in range(n_iters):
        h, _ = chunked_scan(dA, dBx)
        mx.eval(h)
    chunk_time = (time.time() - t0) / n_iters

    speedup = seq_time / chunk_time if chunk_time > 0 else float('inf')
    print(f"  Sequential: {seq_time*1000:.1f}ms | Chunked: {chunk_time*1000:.1f}ms | Speedup: {speedup:.1f}×")
    return speedup


def test_full_model_forward():
    """End-to-end: run AgentMind forward with optimized scan."""
    from config import AgentMindConfig
    from model.agent_lm import AgentMind
    from init import init_agentmind

    cfg = AgentMindConfig()
    cfg.debug = False

    model = AgentMind(cfg)
    model = init_agentmind(model, cfg)

    mx.random.seed(123)
    input_ids = mx.random.randint(0, cfg.vocab_size, shape=(1, 256))

    logits, _ = model(input_ids)
    mx.eval(logits)

    has_nan = not mx.all(mx.isfinite(logits)).item()
    mean_val = mx.mean(logits).item()
    std_val = mx.std(logits).item()

    print(f"  Logits shape {logits.shape}, mean={mean_val:.4f}, std={std_val:.4f}, has_nan={has_nan}")
    
    if has_nan:
        print(f"  ❌ FAIL: Output contains NaN/Inf")
        return False
    
    print(f"  ✅ Forward pass clean — no NaN")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("  Chunked Parallel Scan Parity Tests")
    print("=" * 60)

    all_pass = True

    # Parity tests
    all_pass &= test_parity("Small (B=1,L=8,D=4,N=2)", 1, 8, 4, 2)
    all_pass &= test_parity("Medium (B=1,L=64,D=128,N=16)", 1, 64, 128, 16)
    all_pass &= test_parity("Training (B=1,L=256,D=2048,N=64)", 1, 256, 2048, 64)
    all_pass &= test_parity("With h_init (B=1,L=256,D=2048,N=64)", 1, 256, 2048, 64, use_h_init=True)
    all_pass &= test_parity("Batch (B=2,L=128,D=512,N=32)", 2, 128, 512, 32)

    # Edge cases
    print("\n  Edge cases:")
    mx.random.seed(99)
    dA = mx.random.uniform(shape=(1, 256, 2048, 64), low=0.01, high=0.1)
    dBx = mx.random.normal(shape=(1, 256, 2048, 64)) * 0.5
    h_seq, _ = sequential_scan(dA, dBx)
    h_par, _ = chunked_scan(dA, dBx)
    mx.eval(h_seq, h_par)
    diff = mx.max(mx.abs(h_seq - h_par)).item()
    edge_pass = diff < 1e-3
    print(f"  Aggressive decay (full dim): max_diff={diff:.2e} {'✅ PASS' if edge_pass else '❌ FAIL'}")
    all_pass &= edge_pass

    # Speed benchmark
    print(f"\n  Speed benchmark (B=1, L=256, D=2048, N=64):")
    test_speed(1, 256, 2048, 64)

    # Full model integration
    print(f"\n  Full model integration (L=256):")
    try:
        all_pass &= test_full_model_forward()
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        all_pass = False

    print("\n" + "=" * 60)
    if all_pass:
        print("  🎉 All tests PASSED!")
    else:
        print("  ⚠️  Some tests FAILED")
    print("=" * 60)
