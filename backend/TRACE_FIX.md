# Trace Fix: Nested LLM Spans in Arize

**Date:** February 13, 2026  
**Status:** Fixed  

---

## Problem

Agent LLM calls appeared as orphaned root traces in Arize instead of nesting under `run_agents`:

```
BEFORE (broken):                        AFTER (fixed):

Trace 1: LangGraph Chain                LangGraph Chain
 ├─ fetch_data                           ├─ fetch_data
 ├─ temporal_analysis                    ├─ temporal_analysis
 ├─ run_agents  ← opaque                 ├─ run_agents
 └─ synthesize                           │   ├─ agent.buffett → ChatOpenAI
                                         │   ├─ agent.lynch → ChatOpenAI
Trace 2: ChatOpenAI (orphaned)           │   ├─ agent.graham → ChatOpenAI
Trace 3: ChatOpenAI (orphaned)           │   ├─ agent.munger → ChatOpenAI
                                         │   └─ agent.dalio → ChatOpenAI
                                         └─ synthesize
                                             └─ consensus.synthesis → ChatOpenAI
```

---

## Root Cause

`openinference-instrumentation-langchain` links spans via LangChain's `Run.parent_run_id`, **not** OpenTelemetry `contextvars`. Calling `ChatOpenAI.ainvoke(messages)` without a `RunnableConfig` produces an orphaned root trace.

Additionally, 4/5 agents had bare `except Exception:` blocks silently swallowing LLM errors.

---

## Fix (two parts)

### 1. Thread `RunnableConfig` through the call chain

```
LangGraph node(state, config)
  → agent.analyze(..., config=config)
    → llm_client.analyze(..., config=config)
      → ChatOpenAI.ainvoke(messages, config=config)  ← parent_run_id set
```

**Files:** `llm/client.py`, all 5 `agents/*.py`, `strategies/__init__.py`

### 2. Wrap agents in `RunnableLambda` for named spans

```python
runnable = RunnableLambda(_analyze).with_config({"run_name": f"agent.{name}"})
tasks.append(runnable.ainvoke(None, config=config))
```

This gives each agent a named span in Arize. Same pattern for consensus:

```python
RunnableLambda(_consensus_fn).with_config({"run_name": "consensus.synthesis"})
```

**File:** `main.py` — also simplified agent init with `AGENT_REGISTRY` loop.

---

## Key Lesson

When using `openinference-instrumentation-langchain`, **always pass `RunnableConfig`** from LangGraph nodes to `ChatOpenAI.ainvoke()`. The tracer ignores OTel context — it only uses LangChain's `parent_run_id`. Wrap custom logic in `RunnableLambda` for named spans.
