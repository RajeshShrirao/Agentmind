DOMAIN_CONFIGS = {
    "tool_caller": {
        "hf_datasets": [],
        "synthetic_count": 103703,
        "adversarial_rate": 0.3,
        "prebuilt_path": "data/apprentice_tool_caller.jsonl",
        "diversity": {"multi_tool_pct": 78.0, "plan_pct": 39.6, "avg_tool_calls": 2.6, "tools": 14},
    },
    "planner": {
        "hf_datasets": [
            ("open-thoughts/AgentTrove", None, "train",
             lambda x: (4 < len(x.get("messages", [])) < 20
                        and sum(len(m.get("content", "")) for m in x.get("messages", [])) < 12000),
             3000, {}),
            ("THUDM/AgentInstruct", None, "mind2web",
             lambda x: (1 < len(x.get("conversations", [])) < 15
                        and sum(len(c.get("value", "")) for c in x.get("conversations", [])) < 10000),
             1000, {}),
            ("THUDM/AgentInstruct", None, "webshop",
             lambda x: (1 < len(x.get("conversations", [])) < 15
                        and sum(len(c.get("value", "")) for c in x.get("conversations", [])) < 10000),
             500, {}),
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
            ("bigcode/the-stack", None, "train",
             lambda x: x.get("lang") == "python" and len(x.get("content", "")) < 5000,
             2000, {"data_dir": "data/python"}),
            ("sahil2801/CodeAlpaca-20k", None, "train", None, 3000, {}),
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
