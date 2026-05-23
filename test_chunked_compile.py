import mlx.core as mx
import time

@mx.compile
def scan_chunk(dA_c, dBx_c, h):
    C = dA_c.shape[1]
    h_seq = []
    for t in range(C):
        h = dA_c[:, t] * h + dBx_c[:, t]
        h_seq.append(h)
    return mx.stack(h_seq, axis=1), h

def chunked_compiled_scan(dA, dBx, chunk_size=16):
    B, L, D, N = dA.shape
    h = mx.zeros((B, D, N))
    h_chunks = []
    
    for t0 in range(0, L, chunk_size):
        t1 = min(t0 + chunk_size, L)
        h_chunk, h = scan_chunk(dA[:, t0:t1], dBx[:, t0:t1], h)
        h_chunks.append(h_chunk)
        
    return mx.concatenate(h_chunks, axis=1)

def sequential_scan(dA, dBx):
    B, L, D, N = dA.shape
    h = mx.zeros((B, D, N))
    h_seq = []
    for t in range(L):
        h = dA[:, t] * h + dBx[:, t]
        h_seq.append(h)
    return mx.stack(h_seq, axis=1)

mx.random.seed(99)
dA = mx.random.uniform(shape=(1, 256, 2048, 64), low=0.01, high=0.1)
dBx = mx.random.normal(shape=(1, 256, 2048, 64)) * 0.5

# Compile Warmup
h_chunk = chunked_compiled_scan(dA, dBx)
mx.eval(h_chunk)

h_seq = sequential_scan(dA, dBx)
mx.eval(h_seq)
diff = mx.max(mx.abs(h_seq - h_chunk)).item()
print(f"Diff: {diff:.2e}")

# Benchmark
t0 = time.time()
for _ in range(10):
    mx.eval(sequential_scan(dA, dBx))
seq_time = (time.time() - t0) / 10

t0 = time.time()
for _ in range(10):
    mx.eval(chunked_compiled_scan(dA, dBx))
chunk_time = (time.time() - t0) / 10

print(f"Sequential: {seq_time*1000:.1f}ms")
print(f"Chunked compiled: {chunk_time*1000:.1f}ms")
