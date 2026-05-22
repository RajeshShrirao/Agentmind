import mlx.core as mx
import mlx.nn as nn
from config import AgentMindConfig
from model.mamba_block import MambaBlock
import sys

def test_mamba_parity():
    print("Initializing test configurations...")
    # Instantiate config with smaller dimensions for fast/clean testing
    cfg = AgentMindConfig(
        d_model=64,
        d_state=16,
        d_conv=4,
        expand=2
    )

    mx.random.seed(42)

    # Instantiate MambaBlock
    block = MambaBlock(cfg)

    # Test cases: (B, L)
    test_shapes = [
        (1, 1),
        (2, 5),
        (3, 16),
        (4, 32)
    ]

    for B, L in test_shapes:
        print(f"\n================ Running Test Case: B={B}, L={L} ================")
        x = mx.random.normal((B, L, cfg.d_model))

        # -------------------------------------------------------------
        # Test 1: Full Sequence Forward vs Token-by-Token Recurrence
        # -------------------------------------------------------------
        print("Running full sequence forward pass...")
        out_call, final_state = block(x)

        print("Running token-by-token step recurrence loop...")
        curr_state = None
        step_outputs = []
        for t in range(L):
            token_in = x[:, t, :]
            token_out, curr_state = block.step(token_in, curr_state)
            step_outputs.append(token_out)
        out_step = mx.stack(step_outputs, axis=1)

        # Check Output Parity
        out_diff = mx.max(mx.abs(out_call - out_step)).item()
        print(f"Output maximum absolute difference: {out_diff:.8e}")
        if out_diff > 1e-5:
            print("ERROR: Output parity broken between __call__ and step()!", file=sys.stderr)
            sys.exit(1)

        # Check SSM State Parity
        ssm_diff = mx.max(mx.abs(final_state["ssm_state"] - curr_state["ssm_state"])).item()
        print(f"SSM State maximum absolute difference: {ssm_diff:.8e}")
        if ssm_diff > 1e-5:
            print("ERROR: SSM State parity broken!", file=sys.stderr)
            sys.exit(1)

        # Check Conv State Parity
        conv_diff = mx.max(mx.abs(final_state["conv_state"] - curr_state["conv_state"])).item()
        print(f"Conv State maximum absolute difference: {conv_diff:.8e}")
        if conv_diff > 1e-5:
            print("ERROR: Conv State parity broken!", file=sys.stderr)
            sys.exit(1)

        # -------------------------------------------------------------
        # Test 2: Chunked Forward Pass (State Propagation)
        # -------------------------------------------------------------
        if L > 2:
            print("Running chunked sequence forward pass with state propagation...")
            split_idx = L // 2
            x1 = x[:, :split_idx, :]
            x2 = x[:, split_idx:, :]

            out_chunk1, state1 = block(x1)
            out_chunk2, state2 = block(x2, state1)
            out_chunked = mx.concatenate([out_chunk1, out_chunk2], axis=1)

            # Compare chunked output vs full output
            chunk_out_diff = mx.max(mx.abs(out_call - out_chunked)).item()
            print(f"Chunked output maximum absolute difference: {chunk_out_diff:.8e}")
            if chunk_out_diff > 1e-5:
                print("ERROR: Chunked forward output parity broken!", file=sys.stderr)
                sys.exit(1)

            # Compare final state of chunked vs full
            chunk_ssm_diff = mx.max(mx.abs(final_state["ssm_state"] - state2["ssm_state"])).item()
            chunk_conv_diff = mx.max(mx.abs(final_state["conv_state"] - state2["conv_state"])).item()
            print(f"Chunked SSM State maximum absolute difference: {chunk_ssm_diff:.8e}")
            print(f"Chunked Conv State maximum absolute difference: {chunk_conv_diff:.8e}")
            if chunk_ssm_diff > 1e-5 or chunk_conv_diff > 1e-5:
                print("ERROR: Chunked final state parity broken!", file=sys.stderr)
                sys.exit(1)

        # -------------------------------------------------------------
        # Test 3: Multi-dimensional Input Shapes (Prefix batch dimensions)
        # -------------------------------------------------------------
        print("Testing multi-dimensional/complex batch prefix shapes...")
        # E.g. [2, 3, d_model] for a single step
        x_prefix = mx.random.normal((2, 3, cfg.d_model))
        curr_state_prefix = None
        
        # Test step with 3D input (should squeeze, process, and unsqueeze back)
        out_prefix, curr_state_prefix = block.step(x_prefix, curr_state_prefix)
        if out_prefix.shape != x_prefix.shape:
            print(f"ERROR: step() output shape {out_prefix.shape} does not match input shape {x_prefix.shape}!", file=sys.stderr)
            sys.exit(1)
        
        # Check that state contains correct dimensions
        # Total batch size is 2 * 3 = 6
        expected_batch_size = 6
        if curr_state_prefix["ssm_state"].shape != (expected_batch_size, cfg.d_inner, cfg.d_state):
            print("ERROR: SSM state shape mismatch for prefix input!", file=sys.stderr)
            sys.exit(1)
        if curr_state_prefix["conv_state"].shape != (expected_batch_size, cfg.d_conv - 1, cfg.d_inner):
            print("ERROR: Conv state shape mismatch for prefix input!", file=sys.stderr)
            sys.exit(1)

    print("\nALL tests passed successfully! Parity and state serialization verified.")

if __name__ == "__main__":
    test_mamba_parity()
