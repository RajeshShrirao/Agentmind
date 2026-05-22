import unittest
from tokenizer_setup import load_tokenizer, get_token_ids, assert_token_ids_real, hydrate_config
from config import AgentMindConfig

TOKEN_NAMES = [
    "<pad>", "<s>", "</s>", "<unk>",
    "<bos>", "<eos>",
    "<|tool_call|>", "<|plan|>", "<|memory|>", "<|scratch|>", "<|observe|>",
    "<|think_start|>", "<|think_end|>",
    "<|system|>", "<|user|>", "<|assistant|>",
]

class TestTokenizerIDConsistency(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tok = load_tokenizer("agentmind_tok.model")
        cls.ids = get_token_ids(cls.tok)

    def test_all_special_tokens_have_unique_ids(self):
        all_ids = [
            self.ids.pad_id, self.ids.bos_id, self.ids.eos_id, self.ids.unk_id,
            self.ids.tool_call_id, self.ids.plan_id, self.ids.memory_id, self.ids.scratch_id,
            self.ids.observe_id, self.ids.think_start_id, self.ids.think_end_id,
            self.ids.system_id, self.ids.user_id, self.ids.assistant_id,
        ]
        self.assertEqual(len(set(all_ids)), len(all_ids), "Duplicate token IDs detected")

    def test_standard_spm_ids(self):
        self.assertEqual(self.ids.pad_id, 0)
        self.assertEqual(self.ids.bos_id, 1)
        self.assertEqual(self.ids.unk_id, 3)
        self.assertEqual(self.tok.eos_id(), 2)

    def test_agentic_tokens_have_ids_above_3(self):
        agentic = [
            self.ids.tool_call_id, self.ids.plan_id, self.ids.memory_id, self.ids.scratch_id,
            self.ids.observe_id, self.ids.think_start_id, self.ids.think_end_id,
            self.ids.system_id, self.ids.user_id, self.ids.assistant_id,
            self.ids.eos_id,
        ]
        for tid in agentic:
            self.assertGreater(tid, 3, f"Agentic token ID {tid} should be > 3")

    def test_ids_match_tokenizer_lookup(self):
        self.assertEqual(self.ids.eos_id, self.tok.piece_to_id("<eos>"))
        self.assertEqual(self.ids.tool_call_id, self.tok.piece_to_id("<|tool_call|>"))
        self.assertEqual(self.ids.assistant_id, self.tok.piece_to_id("<|assistant|>"))
        self.assertEqual(self.ids.system_id, self.tok.piece_to_id("<|system|>"))
        self.assertEqual(self.ids.user_id, self.tok.piece_to_id("<|user|>"))


class TestConfigHydration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tok = load_tokenizer("agentmind_tok.model")

    def test_hydration_sets_all_ids(self):
        cfg = AgentMindConfig()
        hydrate_config(cfg, self.tok)
        self.assertEqual(cfg.pad_id, 0)
        self.assertEqual(cfg.bos_id, 1)
        self.assertEqual(cfg.eos_id, 5)
        self.assertEqual(cfg.tool_call_id, 6)
        self.assertEqual(cfg.assistant_id, 15)

    def test_pre_hydration_ids_are_sentinel(self):
        cfg = AgentMindConfig()
        for attr in ("pad_id", "bos_id", "eos_id", "tool_call_id", "plan_id",
                     "memory_id", "scratch_id", "observe_id", "think_start_id",
                     "think_end_id", "system_id", "user_id", "assistant_id"):
            self.assertEqual(getattr(cfg, attr), -1, f"{attr} should be -1 before hydration")

    def test_hydration_preserves_architecture_params(self):
        cfg = AgentMindConfig()
        hydrate_config(cfg, self.tok)
        self.assertEqual(cfg.vocab_size, 32000)
        self.assertEqual(cfg.d_model, 1024)
        self.assertEqual(cfg.n_layers, 16)
        self.assertEqual(cfg.dt_rank_val, 64)


class TestTokenIDRoundtrip(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tok = load_tokenizer("agentmind_tok.model")

    def test_roundtrip_special_tokens(self):
        text = "<|user|>hello<|assistant|>world<eos>"
        ids = self.tok.encode(text, add_bos=True)
        decoded = self.tok.decode(ids)
        self.assertIn("hello", decoded)
        self.assertIn("world", decoded)

    def test_tokenized_ids_match_cfg(self):
        cfg = AgentMindConfig()
        hydrate_config(cfg, self.tok)
        text = "<|user|>hello<|assistant|>world<eos>"
        ids = self.tok.encode(text, add_bos=True)
        self.assertIn(cfg.user_id, ids)
        self.assertIn(cfg.assistant_id, ids)
        self.assertIn(cfg.eos_id, ids)


class TestAllSpecialTokensDecodeable(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tok = load_tokenizer("agentmind_tok.model")
        cls.ids = get_token_ids(cls.tok)

    def test_all_tokens_decodeable(self):
        expected = {
            "pad_id": "<pad>",
            "bos_id": "<s>",
            "eos_id": "<eos>",
            "unk_id": "<unk>",
            "tool_call_id": "<|tool_call|>",
            "plan_id": "<|plan|>",
            "memory_id": "<|memory|>",
            "scratch_id": "<|scratch|>",
            "observe_id": "<|observe|>",
            "think_start_id": "<|think_start|>",
            "think_end_id": "<|think_end|>",
            "system_id": "<|system|>",
            "user_id": "<|user|>",
            "assistant_id": "<|assistant|>",
        }
        for attr, expected_str in expected.items():
            tok_id = getattr(self.ids, attr)
            actual_str = self.tok.id_to_piece(tok_id)
            self.assertEqual(actual_str, expected_str,
                             f"{attr}={tok_id}: expected '{expected_str}', got '{actual_str}'")


if __name__ == "__main__":
    unittest.main()
