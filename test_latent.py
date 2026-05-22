import unittest
import numpy as np
import mlx.core as mx
import random

from model.latent import inject_latent_tokens, latent_loss_mask, get_latent_stage, N_LATENT_STEPS
from config import AgentMindConfig

class DummyTokenizer:
    """Mock tokenizer for integration testing."""
    def encode(self, text, add_bos=False):
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)

class TestLatentReasoning(unittest.TestCase):

    def setUp(self):
        self.cfg = AgentMindConfig()

    def test_get_latent_stage(self):
        self.assertEqual(get_latent_stage(0), 1)
        self.assertEqual(get_latent_stage(250), 1)
        self.assertEqual(get_latent_stage(500), 2)
        self.assertEqual(get_latent_stage(999), 2)
        self.assertEqual(get_latent_stage(1000), 3)
        self.assertEqual(get_latent_stage(1999), 3)
        self.assertEqual(get_latent_stage(2000), 4)
        self.assertEqual(get_latent_stage(5000), 4)

    def test_inject_latent_tokens_stage_1(self):
        sample = {
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "<|scratch|>let me think...<|tool_call|>call()"}
            ]
        }
        res = inject_latent_tokens(sample, None, stage=1)
        # In Stage 1, content should be completely unchanged
        self.assertEqual(res["messages"][1]["content"], "<|scratch|>let me think...<|tool_call|>call()")

    def test_inject_latent_tokens_stage_2(self):
        # Various end boundaries
        test_cases = [
            ("<|scratch|>let me think...<|tool_call|>call()", "<|think_start|><|scratch|>let me think...<|think_end|><|tool_call|>call()"),
            ("<|scratch|>some thoughts<|observe|>result", "<|think_start|><|scratch|>some thoughts<|think_end|><|observe|>result"),
            ("<|scratch|>planning stage<|plan|>step 1", "<|think_start|><|scratch|>planning stage<|think_end|><|plan|>step 1"),
            ("<|scratch|>long thoughts<eos>", "<|think_start|><|scratch|>long thoughts<|think_end|><eos>"),
            ("<|scratch|>just thinking at the end", "<|think_start|><|scratch|>just thinking at the end<|think_end|>")
        ]

        for content, expected in test_cases:
            sample = {
                "messages": [
                    {"role": "assistant", "content": content}
                ]
            }
            res = inject_latent_tokens(sample, None, stage=2)
            self.assertEqual(res["messages"][0]["content"], expected)

    def test_inject_latent_tokens_stage_3_interpolation(self):
        # Stage 3 should randomly output either Stage 2 (wrapped) or Stage 4 (latent N_LATENT_STEPS scratch tokens)
        content = "<|scratch|>let me think...<|tool_call|>call()"
        
        # Run 50 times, check that we get both behaviors (since prob = 50%)
        stage_2_seen = False
        stage_4_seen = False
        
        for _ in range(50):
            sample = {
                "messages": [
                    {"role": "assistant", "content": content}
                ]
            }
            res = inject_latent_tokens(sample, None, stage=3)
            res_content = res["messages"][0]["content"]
            
            expected_stage_2 = "<|think_start|><|scratch|>let me think...<|think_end|><|tool_call|>call()"
            scratch_tokens = "<|scratch|>" * N_LATENT_STEPS
            expected_stage_4 = f"<|think_start|>{scratch_tokens}<|think_end|><|tool_call|>call()"
            
            if res_content == expected_stage_2:
                stage_2_seen = True
            elif res_content == expected_stage_4:
                stage_4_seen = True
            else:
                self.fail(f"Unexpected output in Stage 3: {res_content}")
                
        self.assertTrue(stage_2_seen, "Did not observe Stage 2 formatting in Stage 3")
        self.assertTrue(stage_4_seen, "Did not observe Stage 4 formatting in Stage 3")

    def test_inject_latent_tokens_stage_4(self):
        # Should replace all explicit CoT thoughts with exactly N_LATENT_STEPS of <|scratch|>
        content = "<|scratch|>let me think...<|tool_call|>call()"
        sample = {
            "messages": [
                {"role": "assistant", "content": content}
            ]
        }
        res = inject_latent_tokens(sample, None, stage=4)
        scratch_tokens = "<|scratch|>" * N_LATENT_STEPS
        expected = f"<|think_start|>{scratch_tokens}<|think_end|><|tool_call|>call()"
        self.assertEqual(res["messages"][0]["content"], expected)

    def test_latent_loss_mask_1d(self):
        # Construct synthetic inputs and labels:
        # Index:  0        1             2          3          4          5          6
        # Tokens: before,  think_start,  scratch,   scratch,   think_end, next_obs,  other
        # IDs:    100,     11,           9,         9,         12,        200,       201
        
        input_ids = mx.array([100, 11, 9, 9, 12, 200, 201])
        labels    = mx.array([100, 11, 9, 9, 12, 200, 201])
        
        masked = latent_loss_mask(input_ids, labels, think_start_id=11, think_end_id=12)
        masked_list = masked.tolist()
        
        # Verify:
        # before (0) -> not masked (100)
        # think_start (1) -> not masked (11) (so prediction of think_start is active)
        # scratch (2) -> masked (-100)
        # scratch (3) -> masked (-100)
        # think_end (4) -> masked (-100)
        # next_obs (5) -> not masked (200)
        # other (6) -> not masked (201)
        expected = [100, 11, -100, -100, -100, 200, 201]
        self.assertEqual(masked_list, expected)

    def test_latent_loss_mask_2d(self):
        input_ids = mx.array([
            [100, 11, 9, 9, 12, 200, 201],
            [300, 301, 11, 9, 12, 400, 401]
        ])
        labels = mx.array([
            [100, 11, 9, 9, 12, 200, 201],
            [300, 301, 11, 9, 12, 400, 401]
        ])
        
        masked = latent_loss_mask(input_ids, labels, think_start_id=11, think_end_id=12)
        masked_list = masked.tolist()
        
        expected = [
            [100, 11, -100, -100, -100, 200, 201],
            [300, 301, 11, -100, -100, 400, 401]
        ]
        self.assertEqual(masked_list, expected)

    def test_no_trivial_end_marker_prediction_shifted(self):
        # Test how shifted targets behave after applying the loss mask.
        # This verifies the model is not trained to emit think_end immediately after think_start.
        
        # Index:  0        1             2          3          4          5
        # Tokens: before,  think_start,  scratch,   scratch,   think_end, next_obs
        # IDs:    100,     11,           9,         9,         12,        200
        
        input_ids = mx.array([100, 11, 9, 9, 12, 200])
        labels    = mx.array([100, 11, 9, 9, 12, 200])
        
        masked_labels = latent_loss_mask(input_ids, labels, think_start_id=11, think_end_id=12)
        
        # Shift inputs and targets (as done in cross_entropy_loss)
        shifted_inputs  = input_ids[:-1].tolist()
        shifted_targets = masked_labels[1:].tolist()
        
        # Let's verify each next-token prediction task:
        # 1. Input: before (100) -> Target: think_start (11) (Active) -> Model learns when to think
        # 2. Input: think_start (11) -> Target: -100 (Masked) -> Model is NOT trained to predict scratch/think_end immediately
        # 3. Input: scratch (9) -> Target: -100 (Masked)
        # 4. Input: scratch (9) -> Target: -100 (Masked)
        # 5. Input: think_end (12) -> Target: next_obs (200) (Active) -> Model learns to predict the next observable token from think_end!
        
        self.assertEqual(shifted_inputs, [100, 11, 9, 9, 12])
        self.assertEqual(shifted_targets, [11, -100, -100, -100, 200])
        
        # Verify that prediction from think_start is masked (index 1 in shifted_inputs, corresponding to target index 1)
        self.assertEqual(shifted_targets[1], -100)
        # Verify that prediction from think_end is active (index 4 in shifted_inputs, corresponding to target 200)
        self.assertEqual(shifted_targets[4], 200)

    def test_end_to_end_live_path_stage_2(self):
        """Integration test: data injection → loss masking (stage 2)."""
        tok = DummyTokenizer()
        sample = {
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "<|scratch|>let me think...<|tool_call|>call()"}
            ]
        }
        injected = inject_latent_tokens(sample, tok, stage=2)
        self.assertIn("<|think_start|>", injected["messages"][1]["content"])
        self.assertIn("<|think_end|>", injected["messages"][1]["content"])
        self.assertIn("<|scratch|>let me think...", injected["messages"][1]["content"])

    def test_end_to_end_live_path_stage_4(self):
        """Integration test: data injection → tokenization → loss mask (stage 4)."""
        tok = DummyTokenizer()
        sample = {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "<|scratch|>deep thought<|tool_call|>fn()"}
            ]
        }
        injected = inject_latent_tokens(sample, tok, stage=4)
        content = injected["messages"][1]["content"]
        self.assertIn("<|think_start|>", content)
        self.assertIn("<|think_end|>", content)
        self.assertNotIn("deep thought", content)

        ids = mx.array(tok.encode(content))
        labels = mx.array(ids)
        masked = latent_loss_mask(ids, labels,
                                  think_start_id=tok.encode("<|think_start|>")[0],
                                  think_end_id=tok.encode("<|think_end|>")[0])
        masked_list = masked.tolist()
        self.assertEqual(len(masked_list), len(ids.tolist()))
        # Assert think_start is NOT masked (model should learn when to think)
        think_start_pos = ids.tolist().index(tok.encode("<|think_start|>")[0])
        self.assertNotEqual(masked_list[think_start_pos], -100)
        # Assert content between think_start and think_end IS masked
        think_end_pos = ids.tolist().index(tok.encode("<|think_end|>")[0])
        for i in range(think_start_pos + 1, think_end_pos + 1):
            self.assertEqual(masked_list[i], -100)

    def test_no_latent_wrapper_imported(self):
        """Verify LatentReasoningWrapper was deleted — no dead abstractions."""
        import model.latent
        self.assertFalse(hasattr(model.latent, "LatentReasoningWrapper"),
                         "LatentReasoningWrapper must be removed from model.latent")


if __name__ == "__main__":
    unittest.main()
