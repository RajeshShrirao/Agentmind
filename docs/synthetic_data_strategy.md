# Synthetic Data Generation Strategy

> Hybrid approach: LLM-crafted seeds → programmatic augmentation → multi-layer cognitive diversity → 100K+ agentic trajectories.

## Pipeline Overview

```
  Phase 1                          Phase 2                          Phase 3
  5 subagents (parallel)           augment_tool_caller_data.py      augmentation/ (8 layers)
  ┌──────────────┐                 ┌──────────────────────┐         ┌──────────────────────┐
  │ Agent 1: 200 │                 │  For each seed × 26: │         │  1. Semantic Query   │
  │ Agent 2: 200 │──→ 984 seeds    │  1. Swap entities    │──→ 25K  │  2. Tool Order       │
  │ Agent 3: 200 │   (natural,     │  2. Re-roll args     │   base  │  3. Observation       │
  │ Agent 4: 200 │    all tools)   │  3. Inject failures  │  seeds  │  4. Retry/Recovery    │──→ 100K+
  │ Agent 5: 200 │                 │  4. Update query     │         │  5. Planner Style     │   final
  └──────────────┘                 └──────────────────────┘         │  6. Graph (NetworkX)  │
                                                                    │  7. Adversarial       │
                                                                    │  8. Environment (Faker)│
                                                                    └──────────────────────┘
```

## Phase 1: Seed Generation via Subagents

Spawn N subagents in parallel (5 works well), each writing to a temp file.

**Prompt structure for each agent:**
1. **Tool registry** — all tools with exact parameter names (prevents hallucinated params)
2. **Output format** — exact JSONL schema, one example per variant type
3. **Special tokens** — list them explicitly, warn against nesting inside observe values
4. **Adversarial modes** — enumerate each failure type with exact JSON format
5. **Quality rules** — natural queries, no templated phrases, vary entities

**Diversity trick:** Give each agent a different focus:
- Agent 1: consumer tools (weather, stocks, search, translate)
- Agent 2: developer tools (file ops, git, SQL, code)
- Agent 3: research/communication tools (arxiv, email, fetch)
- Agent 4: heavy multi-tool chains (40%+ multi)
- Agent 5: heavy adversarial patterns (50%+ failures)

**Validation:** Combine temp files, run structural validation (correct domain/type/messages, balanced token counts, valid JSON in tool calls, all 14 tools present).

## Phase 2: Basic Augmentation (25K Base)

**Core idea:** Parse each seed, then generate N variants by changing only what matters — tool args, entities, and failure modes — while preserving the seed's structure (token boundaries, scratch placement, plan text).

**Implementation:**
1. Regex-segment tool calls: `<|tool_call|>(.*?)<|observe|>(.*?)` — capture everything between tokens (not JSON-bounded)
2. For each tool call, generate new args using random entity pools
3. Replace the entire segment by position (not `str.replace()`)
4. For clean seeds: inject adversarial failure on first call at configurable rate (30%)
5. Propagate entity changes back to user query

**Entity pools** (per tool):
- `get_weather`: 24 cities
- `get_stock_price`: 20 tickers
- `send_email`: 7 local × 8 domains = 56 email combos
- `read_file` / `write_file`: 14 file paths
- `list_directory`: 13 directories
- `run_python`: 7 code snippets
- `execute_sql`: 9 SQL queries
- `git_commit`: 10 commit messages
- `search_arxiv`: 10 queries + 10 paper IDs
- `web_search`: 10 queries
- `translate`: 8 text/lang pairs across 8 languages
- `summarize`: 3 texts + 3 summaries

**Critical detail — positional replacement:** Always modify the assistant text by slicing around matched segment positions (`text[:start] + new_segment + text[end:]`), never by `str.replace()`. The latter corrupts samples where the same substring appears in multiple places.

## Phase 3: Multi-Layer Cognitive Augmentation (25K → 100K+)

The augmentation package (`augmentation/`) adds 8 layers of real diversity — not mechanical duplication.

### Directory structure

```
augmentation/
├── __init__.py                  # Package exports
├── core.py                      # Tool registry, parsing, validation, entropy scoring, dedup
├── semantic_mutator.py          # Layer 1 — nlpaug + rule-based paraphrasing
├── trajectory_mutator.py        # Layers 2, 4, 5 — tool order, retry, planner styles
├── observation_mutator.py       # Layer 3 — 12 environment failure modes
├── graph_generator.py           # Layer 6 — NetworkX DAG → linearized trajectories
├── adversarial_mutator.py       # Layer 7 — property-based stress testing (8 strategies)
├── environment_generator.py     # Layer 8 — Faker-powered realistic world states
└── dataset_expander.py          # CLI entrypoint
```

### Layer Descriptions

#### Layer 1 — Semantic Query Mutation
Uses nlpaug for synonym replacement + rule-based transforms:
- **compression**: remove fillers ("could you", "I need")
- **expansion**: add conversational framing
- **indirect**: rephrase as ambiguous ("I'm wondering about...")
- **typo injection**: character-level noise

#### Layer 2 — Tool Order Mutation
For multi-tool trajectories, generates alternative valid execution orders. Maintains logical consistency (search before summarize, read before write).

#### Layer 3 — Observation Mutation
Replaces sterile observations with 12 realistic environment modes:
- `stale_cache`, `timeout`, `partial`, `malformed`, `conflicting`
- `empty`, `permission_denied`, `rate_limit`, `truncated`, `corrupt`

#### Layer 4 — Retry / Recovery Mutation
Injects failure + recovery patterns into clean trajectories:
- Fallback to alternative tool (e.g., `web_search` → `search_arxiv`)
- Retry same tool with modified parameters
- Natural scratch reasoning between failure and retry

#### Layer 5 — Planner Style Mutation
Varies the cognitive style of planning text:
- **cautious**: "Let me verify step by step:"
- **fast executor**: "Running:", "Executing:"
- **recursive decomposer**: "Breaking down:", "Sub-steps:"
- **defensive**: "Safety check —"
- **verbose**: "Detailed plan:\n"
- **minimalist**: "> ", "→ "

#### Layer 6 — Graph-based Generation (NetworkX)
Generates entirely new multi-tool trajectories from dependency DAGs:
- 7 template DAGs (research report, code pipeline, weather email, stock research, data pipeline, deep research, full-stack audit)
- Topological sort for valid execution order
- Parameterized with random entities (topics, cities, tickers, projects)
- Self-loop edges for multi-call same-tool sequences

#### Layer 7 — Adversarial Mutation
Property-based stress testing with 8 strategies:
- Missing required params, typo params, extra unexpected params
- Null values, wrong types, impossible paths, empty strings
- Unicode injection attacks

#### Layer 8 — Environment Simulation (Faker)
Replaces generic observations with realistic data from Faker:
- Web search results with real-looking URLs and snippets
- File contents matching extension type (`.py`, `.json`, `.csv`, `.log`)
- SQL query results matching table type (users, orders, employees)
- Stock data with real-seeming market caps and volumes
- Git commits with real hashes, branches, file counts
- Translation output with proper per-language patterns
- Arxiv results with dates, authors, paper IDs

### Entropy Scoring & Dedup

After all layers generate samples, the pipeline filters:

1. **Structural validation** — rejects malformed samples (~3%)
2. **Duplicate detection** — content hash dedup (~32% drop)
3. **Entropy scoring** — character n-gram + tool diversity + structure bonus; filters degenerate samples below threshold (~9% drop)

## Usage

### Basic augmentation (25K base)

```bash
python augment_tool_caller_data.py --seeds data/apprentice_tool_caller.jsonl --multiplier 26
```

### Multi-layer expansion to 100K+

```bash
# Full pipeline
python -m augmentation.dataset_expander \
  --input data/apprentice_tool_caller.jsonl \
  --target 100000 \
  --output data/apprentice_tool_caller_100k.jsonl

# Dry run to estimate layer contributions
python -m augmentation.dataset_expander --target 100000 --dry-run
```

## Quality Metrics

### Final 100K dataset stats

| Metric | Value |
|---|---|
| Total samples | **103,703** |
| Validation errors | **0** |
| File size | 83.1 MB |
| Multi-tool | 45% |
| Adversarial (`<|scratch|>`) | 46% |
| Planning (`<|plan|>`) | 39% |
| Latent reasoning (`<|think_start|>`) | 12% |
| Avg tool calls | 2.6 (range 1–6) |
| Tools covered | **14/14** |
| Generation time | 11.1s |

### Layer contribution

| Layer | Samples | % of total |
|---|---|---|
| Graph (NetworkX DAGs) | 60,000 | 32% |
| Observation mutation | 30,000 | 16% |
| Environment simulation | 30,000 | 16% |
| Adversarial mutation | 20,000 | 11% |
| Tool order mutation | 6,544 | 3.5% |
| Semantic mutation | 6,357 | 3.4% |
| Planner style | 4,966 | 2.7% |
| Retry / recovery | 2,530 | 1.4% |
| Original seeds (25K base) | 25,584 | 14% |

## Reusing for Other Domains

The same three-phase approach applies to planner, recovery, code, and research:

| Domain | Subagent focus | Layer config changes |
|---|---|---|
| **planner** | Multi-step trajectories, dependency chains | ↑ graph templates, ↑ planner style |
| **recovery** | Cascading failures, state corruption, partial retries | ↑ retry layer, ↑ observation failures |
| **code** | Syntax errors, runtime exceptions, debugging cycles | Swap tool registry, ↑ adversarial |
| **research** | Multi-source synthesis, contradictory findings | ↑ graph templates, ↑ environment simulation |

## Why This Works

- **Quality from LLM seeds** — natural conversational queries, varied phrasings
- **Scale from augmentation** — 1K seeds × 26 × 8 layers → 100K with near-zero API cost
- **Structural integrity** — positional replacement preserves token boundaries
- **Cognitive diversity** — 8 independent mutation axes prevent semantic collapse
- **Realistic environments** — Faker simulation prevents sterile template feel
- **Robustness** — 46% adversarial rate from compounding seed + injection + retry layers
- **Zero validation errors** — proper regex + positional slicing eliminates parsing failures
