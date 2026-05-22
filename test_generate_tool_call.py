import unittest
import mlx.core as mx
from tokenizer_setup import load_tokenizer, hydrate_config
from config import AgentMindConfig
from model.agent_lm import AgentMind


class TestGenerateToolCallE2E(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cfg = AgentMindConfig()
        cls.tok = load_tokenizer("agentmind_tok.model")
        hydrate_config(cls.cfg, cls.tok)

    def test_generate_tool_call_returns_result_dict(self):
        model = AgentMind(self.cfg)
        prompt = "<|user|>Get the weather in Tokyo<|assistant|>"
        ids = mx.array([self.tok.encode(prompt, add_bos=True)])
        from decode import generate_tool_call
        result = generate_tool_call(model, ids, {}, self.cfg, self.tok, max_tokens=20)
        self.assertIn("raw", result)
        self.assertIn("valid", result)
        self.assertIn("tokens", result)
        self.assertIsInstance(result["raw"], str)
        self.assertIsInstance(result["valid"], bool)
        self.assertIsInstance(result["tokens"], list)


if __name__ == "__main__":
    unittest.main()
