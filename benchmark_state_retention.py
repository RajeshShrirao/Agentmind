import mlx.core as mx
import mlx.nn as nn
import math
from config import AgentMindConfig
from model.mamba_block import MambaBlock
from init import init_agentmind

def run_retention_test(d_state, seq_len=2000, prefix_len=10):
    cfg = AgentMindConfig()
    cfg.d_state = d_state
    # Reinitialize dt_rank based on updated config
    cfg.__post_init__()
    cfg.debug = False # disable debug assertions for speed

    # Instantiate block and initialize
    block = MambaBlock(cfg)
    
    class MockModel(nn.Module):
        def __init__(self, block):
            super().__init__()
            self.block = block
            
    model = MockModel(block)
    init_agentmind(model, cfg)

    # Generate two sequences: identical except for the prefix
    mx.random.seed(1337)
    prefix_a = mx.random.normal((1, prefix_len, cfg.d_model))
    prefix_b = mx.random.normal((1, prefix_len, cfg.d_model))
    
    # Same distractors
    distractors = mx.random.normal((1, seq_len - prefix_len, cfg.d_model))
    
    seq_a = mx.concatenate([prefix_a, distractors], axis=1)
    seq_b = mx.concatenate([prefix_b, distractors], axis=1)

    state_a = None
    state_b = None
    
    diffs = []
    rel_diffs = []
    base_diff = None

    for t in range(seq_len):
        token_a = seq_a[:, t, :]
        token_b = seq_b[:, t, :]
        
        # Step recurrence
        _, state_a = block.step(token_a, state_a)
        _, state_b = block.step(token_b, state_b)
        
        ssm_a = state_a["ssm_state"]
        ssm_b = state_b["ssm_state"]
        
        # Measure Euclidean difference
        diff = mx.linalg.norm(ssm_a - ssm_b).item()
        diffs.append(diff)
        
        if t == prefix_len - 1:
            base_diff = diff
            
        if base_diff is not None:
            rel_diffs.append(diff / (base_diff + 1e-8))
        else:
            rel_diffs.append(1.0)
            
    return diffs, rel_diffs

def main():
    print("=" * 60)
    print("      AgentMind SSM State Retention Diagnostic Benchmark      ")
    print("=" * 60)
    print("Testing Mamba SSM memory retention over 2000 steps.")
    print("Comparing d_state = 16 (old) vs d_state = 64 (new upgraded).")
    print("-" * 60)

    seq_len = 2000
    prefix_len = 10
    
    print("Running diagnostic for d_state = 16...")
    diffs_16, rel_diffs_16 = run_retention_test(d_state=16, seq_len=seq_len, prefix_len=prefix_len)
    
    print("Running diagnostic for d_state = 64...")
    diffs_64, rel_diffs_64 = run_retention_test(d_state=64, seq_len=seq_len, prefix_len=prefix_len)
    
    print("\nState retention results (Relative signal remaining vs end of prefix):")
    print(f"{'Step':<10} | {'d_state = 16 (Rel)':<20} | {'d_state = 64 (Rel)':<20} | {'Improvement Factor':<18}")
    print("-" * 75)
    
    # We log at specific steps: 10 (end of prefix), 50, 100, 500, 1000, 2000
    intervals = [10, 50, 100, 500, 1000, 2000]
    for step in intervals:
        idx_rel = step - prefix_len
        if step < prefix_len:
            # still in prefix
            val_16 = 1.0
            val_64 = 1.0
        else:
            val_16 = rel_diffs_16[idx_rel]
            val_64 = rel_diffs_64[idx_rel]
            
        ratio = val_64 / (val_16 + 1e-8)
        print(f"Step {step:<5} | {val_16:<20.6f} | {val_64:<20.6f} | {ratio:.2f}x")
        
    print("-" * 75)
    print("Conclusion:")
    final_ratio = rel_diffs_64[-1] / (rel_diffs_16[-1] + 1e-8)
    print(f"At step 2000, d_state=64 retains {rel_diffs_64[-1]*100:.4f}% of prefix info.")
    print(f"At step 2000, d_state=16 retains {rel_diffs_16[-1]*100:.4f}% of prefix info.")
    print(f"State retention improvement: {final_ratio:.2f}x")
    print("=" * 60)

if __name__ == "__main__":
    main()
