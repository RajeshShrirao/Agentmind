import unittest
import numpy as np
import mlx.core as mx
from model.latent import latent_loss_mask, get_latent_stage, inject_latent_tokens, N_LATENT_STEPS


class TestLatentMaskCorrectness(unittest.TestCase):

    def test_tokens_between_boundaries_masked_1d(self):
        ids = np.array([100, 11, 9, 9, 12, 200])
        labels = np.array([100, 100, 100, 100, 100, 200])
        masked = latent_loss_mask(ids, labels, think_start_id=11, think_end_id=12)
        self.assertEqual(masked[0], 100)
        self.assertEqual(masked[2], -100)
        self.assertEqual(masked[3], -100)
        self.assertEqual(masked[5], 200)

    def test_tokens_outside_boundaries_not_masked_1d(self):
        ids = np.array([11, 9, 9, 12])
        labels = np.array([10, 10, 10, 10])
        masked = latent_loss_mask(ids, labels, think_start_id=11, think_end_id=12)
        self.assertEqual(masked[1], -100)
        self.assertEqual(masked[2], -100)

    def test_no_think_tokens_no_masking(self):
        ids = np.array([100, 200, 300])
        labels = np.array([100, 200, 300])
        masked = latent_loss_mask(ids, labels, think_start_id=11, think_end_id=12)
        self.assertEqual(masked.tolist(), [100, 200, 300])

    def test_multiple_latent_windows(self):
        ids = np.array([100, 11, 1, 12, 200, 11, 2, 12, 300])
        labels = np.array([100, 100, 100, 100, 200, 100, 100, 100, 300])
        masked = latent_loss_mask(ids, labels, think_start_id=11, think_end_id=12)
        expected = [100, 100, -100, -100, 200, 100, -100, -100, 300]
        self.assertEqual(masked.tolist(), expected)


class TestLatentMaskStages(unittest.TestCase):

    def test_stage_1_no_masking(self):
        stage = get_latent_stage(0)
        self.assertEqual(stage, 1)
        ids = np.array([100, 200])
        labels = np.array([100, 200])
        masked = latent_loss_mask(ids, labels, 11, 12)
        self.assertEqual(masked.tolist(), [100, 200])

    def test_stage_2_injects_think_boundaries(self):
        stage = get_latent_stage(500)
        self.assertEqual(stage, 2)
        sample = {
            "messages": [
                {"role": "assistant", "content": "<|scratch|>thinking<|tool_call|>call()"}
            ]
        }
        injected = inject_latent_tokens(sample, None, stage=2)
        content = injected["messages"][0]["content"]
        self.assertIn("<|think_start|>", content)
        self.assertIn("<|think_end|>", content)

    def test_stage_3_interpolates_between_stage_2_and_4(self):
        stage = get_latent_stage(1000)
        self.assertEqual(stage, 3)
        content = "<|scratch|>thinking<|tool_call|>call()"
        stage_2_seen = False
        stage_4_seen = False
        for _ in range(50):
            sample = {"messages": [{"role": "assistant", "content": content}]}
            injected = inject_latent_tokens(sample, None, stage=3)
            c = injected["messages"][0]["content"]
            if "<|think_start|><|scratch|>thinking<|think_end|>" in c:
                stage_2_seen = True
            scratch_tokens = "<|scratch|>" * N_LATENT_STEPS
            if f"<|think_start|>{scratch_tokens}<|think_end|>" in c:
                stage_4_seen = True
        self.assertTrue(stage_2_seen, "Stage 3 never produced stage-2-style output")
        self.assertTrue(stage_4_seen, "Stage 3 never produced stage-4-style output")

    def test_stage_4_removes_cot_content(self):
        stage = get_latent_stage(2000)
        self.assertEqual(stage, 4)
        sample = {
            "messages": [
                {"role": "assistant", "content": "<|scratch|>deep reasoning<|tool_call|>fn()"}
            ]
        }
        injected = inject_latent_tokens(sample, None, stage=4)
        content = injected["messages"][0]["content"]
        self.assertNotIn("deep reasoning", content)
        self.assertIn("<|think_start|>", content)
        self.assertIn("<|think_end|>", content)


if __name__ == "__main__":
    unittest.main()
