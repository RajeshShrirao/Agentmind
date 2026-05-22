"""
JSONL schema definitions for AgentMind training data.
Every training sample is one of five types.
"""

INSTRUCTION_SCHEMA = {
    "type": "instruction",
    "messages": [
        {"role": "system", "content": "You are a precise assistant."},
        {"role": "user",   "content": "Summarize this in 3 bullets: ..."},
        {"role": "assistant", "content": "• Point one\n• Point two\n• Point three"}
    ]
}

TOOL_SINGLE_SCHEMA = {
    "type": "tool_single",
    "messages": [
        {"role": "user", "content": "What is the weather in Pune?"},
        {"role": "assistant", "content": "<|tool_call|>{\"name\": \"get_weather\", \"args\": {\"city\": \"Pune\"}}<|observe|>{\"temp\": 34, \"condition\": \"sunny\"}The weather in Pune is 34°C and sunny."}
    ]
}

AGENT_MULTI_SCHEMA = {
    "type": "agent_multi",
    "messages": [
        {"role": "user", "content": "Find the top AI paper from last week and summarize it."},
        {"role": "assistant", "content": "<|plan|>1. Search arxiv\n2. Fetch abstract\n3. Summarize<|tool_call|>{\"name\": \"search_arxiv\", \"args\": {\"query\": \"AI\", \"days\": 7}}<|observe|>{\"results\": [{\"id\": \"2405.1234\", \"title\": \"Mamba-2\"}]}<|tool_call|>{\"name\": \"fetch_abstract\", \"args\": {\"id\": \"2405.1234\"}}<|observe|>{\"abstract\": \"...\"}\nMamba-2 introduces structured state spaces..."}
    ]
}

RECOVERY_SCHEMA = {
    "type": "recovery",
    "messages": [
        {"role": "user", "content": "Get stock price of NVDA"},
        {"role": "assistant", "content": "<|tool_call|>{\"name\": \"get_stock_price\", \"args\": {\"ticker\": \"NVDA\"}}<|observe|>{\"error\": \"rate_limit\", \"retry_after\": 2}<|scratch|>Tool failed. Retry with backoff.<|tool_call|>{\"name\": \"get_stock_price\", \"args\": {\"ticker\": \"NVDA\", \"source\": \"backup\"}}<|observe|>{\"price\": 1024.5}NVDA is trading at $1024.50."}
    ]
}

VALID_TYPES = {"instruction", "tool_single", "agent_multi", "recovery"}

def validate_sample(sample: dict) -> tuple[bool, str]:
    """
    Validate a training sample against the JSONL schema.
    Returns (is_valid, error_message).
    """
    if not isinstance(sample, dict):
        return False, "Sample must be a dict"

    if "type" not in sample:
        return False, "Missing 'type' field"

    if sample["type"] not in VALID_TYPES:
        return False, f"Invalid type '{sample['type']}', must be one of {VALID_TYPES}"

    if "messages" not in sample:
        return False, "Missing 'messages' field"

    if not isinstance(sample["messages"], list) or len(sample["messages"]) == 0:
        return False, "'messages' must be a non-empty list"

    for i, msg in enumerate(sample["messages"]):
        if not isinstance(msg, dict):
            return False, f"Message {i} must be a dict"
        if "role" not in msg:
            return False, f"Message {i} missing 'role'"
        if "content" not in msg:
            return False, f"Message {i} missing 'content'"
        if not isinstance(msg["content"], str) or len(msg["content"]) == 0:
            return False, f"Message {i} 'content' must be a non-empty string"

    return True, ""
