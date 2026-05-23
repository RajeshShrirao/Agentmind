"""
Generate tool_caller training data via zai-glm-4.7 batching (15 samples/req).
Rate limit: 5 reqs/min → ~75 samples/min.
Target: 3000 samples (~40 min total).
"""

import os, json, re, time, sys
from cerebras.cloud.sdk import Cerebras

TARGET = 3000
BATCH_SIZE = 15
MODEL = "zai-glm-4.7"

TOOL_LISTING = """- web_search(query: string required, max_results: int optional): Search the web
- read_file(path: string required): Read a file from disk
- write_file(path: string required, content: string required): Write content to a file
- run_python(code: string required): Execute Python code in a sandbox
- get_weather(city: string required): Get current weather for a city
- search_arxiv(query: string required, days: int optional): Search arxiv papers
- fetch_abstract(id: string required): Fetch paper abstract by arxiv ID
- execute_sql(query: string required): Run a SQL query
- send_email(to: string required, subject: string required, body: string required): Send an email
- git_commit(message: string required): Commit staged changes
- list_directory(path: string required): List files in a directory
- get_stock_price(ticker: string required): Get current stock price
- translate(text: string required, target_lang: string required): Translate text
- summarize(text: string required): Summarize long text"""

SYSTEM_PROMPT = f"""You are generating training data for an AI assistant that calls tools.

Available tools:
{TOOL_LISTING}

Output exactly {BATCH_SIZE} valid JSON objects, one per line, no other text.

Format per line:
{{"domain": "tool_caller", "type": "tool_single", "messages": [{{"role": "user", "content": "NATURAL user query"}}, {{"role": "assistant", "content": "<|tool_call|>{{"name": "tool_name", "args": {{"param": "value"}}}}<|observe|>{{"result": "output"}}\\nNatural response here."}}]}}

CRITICAL:
- Parameter names MUST match exactly (e.g. "city" not "location" for get_weather)
- User queries must sound like a real person: "What's the weather in Tokyo?" not "Use get_weather for Tokyo"
- VARY tools across all {BATCH_SIZE} samples
- 3-4 samples should have tool failure with <|scratch|> recovery pattern
- 3-4 samples should chain 2+ tools in sequence
- Observe content must be valid JSON
- Natural response text goes after \\n following the observe JSON
- VARY user queries: weather, stocks, email, code, research, file ops, translation, SQL, git, web search, multi-part requests"""

USER_PROMPT = f"""Generate {BATCH_SIZE} diverse tool_caller training samples. 
Use different tools across samples.
Include some failures with <|scratch|> recovery.
Include some multi-tool chains.
Output ONLY {BATCH_SIZE} JSON lines."""


def call_batch(client) -> list[str]:
    """Call API and return list of raw text lines."""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ],
        max_completion_tokens=32768,
        temperature=0.9,
        top_p=0.95,
    )
    text = resp.choices[0].message.content
    # Split into lines, clean up
    lines = text.strip().split("\n")
    return [l.strip() for l in lines if l.strip()]


TOOL_NAMES = {
    "web_search", "read_file", "write_file", "run_python", "get_weather",
    "search_arxiv", "fetch_abstract", "execute_sql", "send_email", "git_commit",
    "list_directory", "get_stock_price", "translate", "summarize",
}


def validate_line(line: str) -> dict | None:
    """Parse and validate a single JSON line."""
    try:
        sample = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(sample, dict):
        return None
    if sample.get("domain") != "tool_caller":
        return None
    if sample.get("type") not in ("tool_single", "tool_multi"):
        return None
    msgs = sample.get("messages")
    if not isinstance(msgs, list) or len(msgs) < 2:
        return None
    for msg in msgs:
        if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
            return None
        if not isinstance(msg["content"], str) or len(msg["content"]) < 3:
            return None
    asst = msgs[-1]["content"]
    if "<|tool_call|>" not in asst or "<|observe|>" not in asst:
        return None
    calls = re.findall(r'<\|tool_call\|>(.*?)<\|observe\|>', asst, re.DOTALL)
    if not calls:
        return None
    for c in calls:
        c = c.strip()
        if not c.startswith("{"):
            return None
        try:
            p = json.loads(c)
        except json.JSONDecodeError:
            return None
        if "name" not in p or "args" not in p:
            return None
        if p["name"] not in TOOL_NAMES:
            return None
    return sample


def main():
    client = Cerebras(api_key=os.environ.get("CEREBRAS_API_KEY", "csk-c2fe5k5hk529dxdd3f4px8m8tct8dmfec8m689cmkp2pwr6m"))

    # Load existing samples
    samples = []
    if os.path.exists("data/apprentice_tool_caller.jsonl"):
        with open("data/apprentice_tool_caller.jsonl") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        print(f"Loaded {len(samples)} existing samples")

    total = len(samples)
    req_count = 0
    invalid_lines = 0
    t0 = time.time()

    print(f"Target: {TARGET} samples, Batch: {BATCH_SIZE}/req, Model: {MODEL}")
    print()

    while total < TARGET:
        req_count += 1
        t_req = time.time()
        lines = []
        try:
            lines = call_batch(client)
        except Exception as e:
            print(f"  ⚠️  Request failed: {e}")
            time.sleep(12)
            continue

        valid = []
        for line in lines:
            sample = validate_line(line)
            if sample:
                valid.append(sample)
            else:
                invalid_lines += 1

        samples.extend(valid)
        total += len(valid)
        elapsed = time.time() - t0
        rate = total / max(elapsed, 1) * 60  # samples per minute

        # Write incrementally
        with open("data/apprentice_tool_caller.jsonl", "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")

        req_time = time.time() - t_req
        print(f"  Req #{req_count}: {len(valid)}/{len(lines)} valid (total: {total}/{TARGET}, {rate:.0f}/min, {req_time:.1f}s)")

        # Rate limit: 5 reqs/min → sleep to stay under
        if total >= TARGET:
            break

        # If request was fast, sleep to respect 5/min limit
        elapsed_since_start = time.time() - t_req
        min_interval = 12.0  # 5 reqs/min = 12s between requests
        if elapsed_since_start < min_interval:
            time.sleep(min_interval - elapsed_since_start)

    elapsed = time.time() - t0
    print(f"\nDone: {total} samples in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Requests: {req_count}, Invalid lines skipped: {invalid_lines}")

    # Stats
    with_scratch = sum(1 for s in samples if "<|scratch|>" in str(s["messages"]))
    multi_tool = sum(1 for s in samples if s["messages"][-1]["content"].count("<|tool_call|>") > 1)
    print(f"  With <|scratch|>: {with_scratch}")
    print(f"  Multi-tool chains: {multi_tool}")


if __name__ == "__main__":
    main()
