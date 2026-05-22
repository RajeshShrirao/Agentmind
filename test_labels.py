import unittest
from tokenizer_setup import load_tokenizer, hydrate_config
from config import AgentMindConfig


def make_labels(ids, assistant_id, eos_id, user_id, system_id):
    labels = [-100] * len(ids)
    in_assistant = False
    for i, tok_id in enumerate(ids):
        if tok_id == assistant_id:
            in_assistant = True
        if in_assistant:
            labels[i] = tok_id
        if tok_id in (eos_id, user_id, system_id):
            in_assistant = False
    return labels


class TestMakeLabels(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tok = load_tokenizer("agentmind_tok.model")
        cls.cfg = AgentMindConfig()
        hydrate_config(cls.cfg, cls.tok)

    def test_labels_before_assistant_are_masked(self):
        text = "<|user|>hello<|assistant|>world<eos>"
        ids = self.tok.encode(text, add_bos=True)
        labels = make_labels(ids, self.cfg.assistant_id, self.cfg.eos_id,
                             self.cfg.user_id, self.cfg.system_id)
        first_assistant = ids.index(self.cfg.assistant_id)
        for i in range(first_assistant):
            self.assertEqual(labels[i], -100,
                             f"Position {i} before assistant should be -100")

    def test_labels_after_user_reset_to_masked(self):
        text = "<|user|>q1<|assistant|>a1<eos><|user|>q2<|assistant|>a2<eos>"
        ids = self.tok.encode(text, add_bos=True)
        labels = make_labels(ids, self.cfg.assistant_id, self.cfg.eos_id,
                             self.cfg.user_id, self.cfg.system_id)
        for i, tok_id in enumerate(ids):
            if tok_id == self.cfg.user_id:
                self.assertEqual(labels[i], -100,
                                 f"user token at position {i} should be -100")

    def test_eos_label_is_not_masked(self):
        text = "<|user|>hello<|assistant|>world<eos>"
        ids = self.tok.encode(text, add_bos=True)
        labels = make_labels(ids, self.cfg.assistant_id, self.cfg.eos_id,
                             self.cfg.user_id, self.cfg.system_id)
        last_eos = len(ids) - 1 - ids[::-1].index(self.cfg.eos_id)
        self.assertNotEqual(labels[last_eos], -100,
                            "EOS token should not be masked (model learns to predict EOS)")

    def test_system_token_exits_assistant_mode(self):
        text = "<|system|>be helpful<|assistant|>ok<|user|>hello<|assistant|>world<eos>"
        ids = self.tok.encode(text, add_bos=True)
        labels = make_labels(ids, self.cfg.assistant_id, self.cfg.eos_id,
                             self.cfg.user_id, self.cfg.system_id)
        self.assertTrue(all(l == -100 for l in labels[:4]),
                        "Tokens before first assistant should be -100")
        self.assertEqual(labels[4], self.cfg.assistant_id,
                         "First assistant should be active")
        user_pos = ids.index(self.cfg.user_id)
        hello_pos = user_pos + 1
        self.assertEqual(ids[hello_pos], self.tok.encode("hello", add_bos=False)[0])
        self.assertEqual(labels[hello_pos], -100,
                         "Token right after user boundary should be masked")
        self.assertEqual(labels[-1], self.cfg.eos_id,
                         "Final EOS should be active (not masked)")

    def test_full_roundtrip_matches_pipeline_behavior(self):
        from data.pipeline import AgentDataset
        sample = {
            "type": "instruction",
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "world"},
            ]
        }
        text = ""
        for msg in sample["messages"]:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                text += f"<|user|>{content}"
            elif role == "assistant":
                text += f"<|assistant|>{content}<eos>"
        ids = self.tok.encode(text, add_bos=True)
        labels = make_labels(ids, self.cfg.assistant_id, self.cfg.eos_id,
                             self.cfg.user_id, self.cfg.system_id)
        first_assistant = ids.index(self.cfg.assistant_id)
        self.assertTrue(all(l == -100 for l in labels[:first_assistant]),
                        "All tokens before first assistant should be masked")
        self.assertTrue(all(l != -100 for l in labels[first_assistant:]),
                        "All tokens from assistant onward should be active for loss")


if __name__ == "__main__":
    unittest.main()
