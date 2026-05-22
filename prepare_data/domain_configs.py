DOMAIN_CONFIGS = {
    "tool_caller": {
        "hf_datasets": [
            ("lambda/hermes-agent-reasoning-traces", "kimi", "train", lambda x: len(x.get("conversations", [])) > 2, 3000, {}),
            ("lambda/hermes-agent-reasoning-traces", "glm-5.1", "train", lambda x: len(x.get("conversations", [])) > 2, 3000, {}),
            ("THUDM/AgentInstruct", None, "os", lambda x: len(x.get("conversations", [])) > 1, 2000, {}),
        ],
        "synthetic_count": 20000,
        "adversarial_rate": 0.3,
    },
    "planner": {
        "hf_datasets": [
            ("open-thoughts/AgentTrove", None, "train", lambda x: len(x.get("messages", [])) > 4, 5000, {}),
            ("THUDM/AgentInstruct", None, "mind2web", lambda x: len(x.get("conversations", [])) > 1, 2000, {}),
            ("THUDM/AgentInstruct", None, "webshop", lambda x: len(x.get("conversations", [])) > 1, 1000, {}),
        ],
        "synthetic_count": 25000,
        "adversarial_rate": 0.3,
    },
    "recovery": {
        "hf_datasets": [],
        "synthetic_count": 30000,
        "adversarial_rate": 0.4,
    },
    "code": {
        "hf_datasets": [
            ("bigcode/the-stack", None, "train", lambda x: x.get("lang") == "python", 10000, {"data_dir": "data/python"}),
            ("sahil2801/CodeAlpaca-20k", None, "train", None, 5000, {}),
        ],
        "synthetic_count": 15000,
        "adversarial_rate": 0.3,
    },
    "research": {
        "hf_datasets": [
            ("HuggingFaceFW/fineweb", "sample-10BT", "train", lambda x: len(x.get("text", "")) > 200, 10000, {}),
            ("HuggingFaceH4/ultrachat_200k", None, "train_sft", None, 5000, {}),
        ],
        "synthetic_count": 20000,
        "adversarial_rate": 0.3,
    },
}
