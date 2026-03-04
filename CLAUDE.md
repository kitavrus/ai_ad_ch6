# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CLI chatbot with OpenAI-compatible APIs featuring a three-tier memory system, pluggable context strategies, and persistent session management.

## Commands

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run
python script.py                          # New session
python script.py --resume                 # Resume last session
python script.py -m "gpt-4" -T 0.8 --strategy sticky_facts

# Test
python -m pytest tests/ -v
python -m pytest tests/test_memory.py -v  # Single module
python -m pytest tests/ -k "memory" -v   # Filter by keyword
python -m pytest tests/ --cov=chatbot     # With coverage
```

Requires `API_KEY` set in `.env` (see `.env.example`).

## Architecture

### Package Structure

```
script.py          # Entry point (imports chatbot.main)
chatbot/
  main.py          # Dialogue loop, command dispatch, API calls (722 LOC)
  cli.py           # argparse setup + inline /command parsing
  config.py        # Constants + SessionConfig (Pydantic)
  models.py        # All Pydantic models (ChatMessage, DialogueSession, etc.)
  memory.py        # Three-tier memory classes + unified Memory facade
  memory_storage.py # Persistence for all three memory tiers
  storage.py       # Session file + per-request metrics logging
  context.py       # Three context strategy implementations
```

### Three-Tier Memory System

`Memory` class unifies three independent stores:

| Tier | Class | Scope | Storage path |
|------|-------|-------|--------------|
| Short-term | `ShortTermMemory` | Current session messages | `dialogues/memory/short_term/` |
| Working | `WorkingMemory` | Current task/preferences | `dialogues/memory/working/` |
| Long-term | `LongTermMemory` | User profile, decisions | `dialogues/memory/long_term/` |

### Context Strategies

Three pluggable strategies selected via `--strategy` or `/strategy` inline command:

- **sliding_window** (default): Keeps last N messages verbatim; older messages are LLM-summarized
- **sticky_facts**: Extracts and prepends key facts as a system message; managed with `/setfact`/`/delfact`
- **branching**: Creates checkpoints (`/checkpoint`) and independent branches (`/branch`, `/switch`)

### Session Persistence

Two independent persistence layers:
1. **Session file**: `dialogues/session_<timestamp>_<model>.json` — full state (messages, strategy, facts, branches)
2. **Per-request metrics**: `dialogues/metrics/session_<id>_req_<idx>.log` — TTFT, tokens, cost in RUB

### Inline Commands (during dialogue)

Format: `/command value` or `/command=value`

Model params: `/temperature`, `/max-tokens`, `/model`
Memory: `/settask`, `/remember`, `/memshow`
Strategy: `/strategy`, `/checkpoint`, `/branch`, `/switch`, `/showfacts`, `/setfact`, `/delfact`
Session: `/resume`, `/showsummary`

### Key Design Patterns

- **All domain objects use Pydantic v2** — type validation, serialization, field constraints (e.g., `temperature ∈ [0, 2]`)
- **API integration**: `openai.OpenAI` client with configurable `base_url` (default: `https://routerai.ru/api/v1`); `top_k` passed via `extra_body`
- **Context building** always delegates to a strategy function in `context.py`; `main.py` only orchestrates
- **Metrics** are computed from wall-clock timestamps and logged independently from session state

### Default Configuration (from `config.py`)

| Constant | Default |
|----------|---------|
| `DEFAULT_MODEL` | `inception/mercury-coder` |
| `DEFAULT_TEMPERATURE` | `0.7` |
| `CONTEXT_RECENT_MESSAGES` | `10` |
| `CONTEXT_SUMMARY_INTERVAL` | `10` |
| `USD_PER_1K_TOKENS` | `0.0015` |
| `RUB_PER_USD` | `100.0` |
