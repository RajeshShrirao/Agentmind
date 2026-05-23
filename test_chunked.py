import mlx.core as mx

def chunked_scan_stable(dA, dBx, h_init=None, chunk_size=32):
    B, L, D, N = dA.shape
    h = h_init if h_init is not None else mx.zeros((B, D, N))
    h_chunks = []
    
    for t0 in range(0, L, chunk_size):
        t1 = min(t0 + chunk_size, L)
        dA_c = dA[:, t0:t1]
        dBx_c = dBx[:, t0:t1]
        
        # Exact product, no log-space mapping needed for short chunks
        cum_dA_c = mx.cumprod(dA_c, axis=1)
        
        # To avoid division by zero if dA has exactly 0, use a small epsilon
        inv_cum_c = 1.0 / mx.maximum(cum_dA_c, 1e-12)
        
        h_chunk = cum_dA_c * mx.cumsum(dBx_c * inv_cum_c, axis=1)
        h_chunk = h_chunk + cum_dA_c * h[:, None, :, :]
        
        h_chunks.append(h_chunk)
        h = h_chunk[:, -1]
        
    h_stack = mx.concatenate(h_chunks, axis=1)
    return h_stack

def sequential_scan(dA, dBx, h_init=None):
    B, L, D, N = dA.shape
    h = h_init if h_init is not None else mx.zeros((B, D, N))
    h_seq = []
    for t in range(L):
        h = dA[:, t] * h + dBx[:, t]
        h_seq.append(h)
    return mx.stack(h_seq, axis=1)

mx.random.seed(99)
dA = mx.random.uniform(shape=(1, 256, 2048, 64), low=0.01, high=0.1)
dBx = mx.random.normal(shape=(1, 256, 2048, 64)) * 0.5
h_seq = sequential_scan(dA, dBx)
h_par = chunked_scan_stable(dA, dBx, chunk_size=16)

mx.eval(h_seq, h_par)
diff = mx.max(mx.abs(h_seq - h_par)).item()
print(f"Diff: {diff:.2e}")
