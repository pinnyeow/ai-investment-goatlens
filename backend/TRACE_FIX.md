# Trace Fix: Orphaned LLM Spans in Arize

**Date:** February 10, 2026  
**Status:** Fixed  
**Files Changed:** 7

---

## Problem

Every GOATlens analysis produced **3 disconnected root traces** in Arize instead of 1 unified trace:

```
BEFORE (broken):
┌─────────────────────────────────────────────┐
│ Trace 1: LangGraph Chain (main workflow)    │
│  ├─ fetch_data_node                         │
│  ├─ temporal_analysis_node                  │
│  ├─ run_agents_node                         │
│  └─ synthesize_node                         │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│ Trace 2: ChatOpenAI (orphaned)              │  ← Buffett's LLM call
│  └─ gpt-4o                                  │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│ Trace 3: ChatOpenAI (orphaned)              │  ← Consensus LLM call
│  └─ gpt-4o-mini                             │
└─────────────────────────────────────────────┘
```

Only 2 LLM calls appeared (not 6) because 4 of 5 agents were silently failing
their `_generate_llm_insights()` calls and falling back to rule-based insights.

---

## Root Cause

### Why spans were orphaned

The `openinference-instrumentation-langchain` tracer does **NOT** use
OpenTelemetry's `contextvars` for parent-child span linking. Instead, it relies
exclusively on LangChain's internal `Run.parent_run_id` mechanism.

From `openinference.instrumentation.langchain._tracer` (lines 170–174):

```python
parent_context = (
    trace_api.set_span_in_context(parent)
    if (parent_run_id := run.parent_run_id)
    and (parent := self._spans_by_run.get(parent_run_id))
    else (context_api.Context() if self._separate_trace_from_runtime_context else None)
)
```

When our agents called `ChatOpenAI.ainvoke(messages)` **without passing a
`RunnableConfig`**, no `parent_run_id` was set on the resulting `Run` object.
The tracer then created a brand new root context — an orphaned trace.

### Why the first fix (asyncio.gather) didn't work

The initial hypothesis was that `asyncio.create_task()` was breaking OTel
context propagation across task boundaries. We switched to passing coroutines
directly to `asyncio.gather()`. This didn't help because the tracer ignores
the OTel runtime context entirely — it only looks at `parent_run_id`.

### Why 4 agents weren't making LLM calls

All 5 agents had `_generate_llm_insights()` methods, but the 4 non-Buffett
agents had bare `except Exception:` blocks that silently swallowed errors.
The actual failures were likely due to missing or malformed arguments in the
early implementation. Without logging, this was invisible.

---

## Fix: Thread `RunnableConfig` Through the Call Chain

LangGraph automatically passes a `RunnableConfig` to each node function as the
second argument. This config contains the callback handlers and run metadata
needed for the tracer to set `parent_run_id` on child runs.

The fix threads this config through the entire call chain:

```
LangGraph node(state, config)        ← LangGraph provides config
  → agent.analyze(..., config=config)
    → _generate_llm_insights(..., config=config)
      → llm_client.analyze(..., config=config)
        → ChatOpenAI.ainvoke(messages, config=config)   ← parent_run_id now set!
```

### After fix:

```
AFTER (unified):
┌───────────────────────────────────────────────────────────┐
│ Trace 1: LangGraph Chain                                  │
│  ├─ fetch_data_node                                       │
│  ├─ temporal_analysis_node                                │
│  ├─ run_agents_node                                       │
│  │   ├─ ChatOpenAI (Buffett, gpt-4o)                     │
│  │   ├─ ChatOpenAI (Lynch, gpt-4o-mini)                  │
│  │   ├─ ChatOpenAI (Graham, gpt-4o-mini)                 │
│  │   ├─ ChatOpenAI (Munger, gpt-4o-mini)                 │
│  │   └─ ChatOpenAI (Dalio, gpt-4o-mini)                  │
│  └─ synthesize_node                                       │
│      └─ ChatOpenAI (Consensus, gpt-4o-mini)              │
└───────────────────────────────────────────────────────────┘
```

All 7 LLM spans (5 agents + 1 consensus + the chain itself) appear as a single
unified trace in Arize.

---

## Files Changed

| File | Change |
|------|--------|
| `llm/client.py` | Added `config=None` param to `analyze()`, passes to `ainvoke()` |
| `main.py` | Added `config` param to `run_agents_node()` and `synthesize_node()`, threads to agents and consensus |
| `agents/buffett.py` | Added `config` to `analyze()` and `_generate_llm_insights()`, added error logging |
| `agents/lynch.py` | Same as buffett |
| `agents/graham.py` | Same as buffett |
| `agents/munger.py` | Same as buffett |
| `agents/dalio.py` | Same as buffett |
| `strategies/__init__.py` | Added `config` to `calculate_consensus_with_llm()`, added error logging |

---

## How to Verify in Arize

1. Run an analysis: `POST /api/analyze {"ticker": "AAPL"}`
2. Open Arize → Project "goatlens"
3. You should see **1 trace** (not 3) per analysis
4. Expanding the trace should show the LangGraph chain with nested LLM spans
5. Check that **6 ChatOpenAI spans** appear nested (5 agents + 1 consensus)
6. If you still see orphaned traces, check server logs for `LLM insight generation failed` messages

---

## Key Lesson

When using `openinference-instrumentation-langchain`, **always pass
`RunnableConfig`** from LangGraph nodes to any `ChatOpenAI.ainvoke()` call.
The tracer's span linking depends on LangChain's callback system, not
OpenTelemetry's context propagation. Without the config, LLM calls become
orphaned root traces.
