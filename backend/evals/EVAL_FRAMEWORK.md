# GOATlens Evaluation Framework

## Why We Need Evals

GOATlens shows financial analysis to retail investors. Two things can go wrong:

1. **Bad data** — If EPS numbers are wrong, beat/miss labels are flipped, or metrics contain NaN values, users see garbage. This is a trust-killer.
2. **Bad insights** — If agents hallucinate numbers, sound generic instead of like their legendary investor, or produce vague platitudes, users have no reason to use GOATlens over ChatGPT.

Evals are how we catch these problems *before* users do.

---

## The 3 Evaluation Dimensions

We chose these 3 dimensions because they map directly to what makes GOATlens valuable (or not) to users.

### Dimension 1: Financial Grounding (Faithfulness)

**What it checks:** Every claim in an insight must be traceable to actual financial data.

**Why it's #1:** In a financial product, hallucinated numbers are dangerous. If Buffett's agent says "ROE of 25%" but the data shows 12%, a user might make a real investment decision based on a lie. This is both a trust issue and a potential legal issue.

**Arize category:** Faithfulness / Hallucination detection

**Method:** LLM-as-Judge with binary classification

**Eval prompt:**
```
You are evaluating a financial analysis tool. You will be given:
1. The raw financial metrics the agent received
2. The insight or concern the agent produced

Determine whether the insight is GROUNDED in the provided metrics,
or whether it introduces claims not supported by the data.

[BEGIN DATA]
************
[Agent]: {agent_name}
[Metrics]: {metrics_json}
[Insight]: {insight_text}
[END DATA]

Consider:
- Does the insight reference specific numbers? If so, do they match the metrics?
- Does the insight make claims about trends or conditions not present in the data?
- Is the directional assessment correct? (e.g., if ROE is 8%, calling it "strong" is wrong)

Your response must be a single word: "grounded" or "hallucinated",
followed by a brief explanation.
```

**Rails:** `grounded` | `hallucinated`

---

### Dimension 2: Investment Philosophy Adherence

**What it checks:** Each agent's output should sound like and reason like their namesake investor.

**Why it's #2:** The entire value proposition of GOATlens is 5 *different* perspectives. If Buffett's agent talks about momentum trading, or Lynch's agent ignores PEG ratios, the product is just 5 copies of the same generic analysis. This dimension protects the core product differentiator.

**Arize category:** Relevance + Tone

**Method:** LLM-as-Judge with binary classification

**Eval prompt:**
```
You are evaluating whether an investment analysis agent stays true
to its assigned investment philosophy.

[BEGIN DATA]
************
[Agent]: {agent_name}
[Philosophy]: {philosophy_description}
[Key Principles]: {principles_list}
[Insight]: {insight_text}
[Concern]: {concern_text}
[END DATA]

Determine whether the analysis reflects the named investor's
known philosophy and principles.

Examples of IN-CHARACTER for Warren Buffett:
- Discussing moats, ROE, owner earnings, margin of safety
- Long-term language ("over decades", "durable advantage")

Examples of OUT-OF-CHARACTER for Warren Buffett:
- Discussing momentum, technical patterns, short-term trading
- Ignoring debt levels or management quality

Your response must be a single word: "in_character" or "out_of_character",
followed by a brief explanation.
```

**Rails:** `in_character` | `out_of_character`

---

### Dimension 3: Actionable Insight Quality

**What it checks:** Would a retail investor find this insight specific enough to reason about?

**Why it's #3:** Users come to GOATlens to make decisions. "This company has potential" tells them nothing. "ROE of 28% has been consistent for 5 years, indicating a durable moat" gives them something to think about. Vague insights = users don't come back.

**Arize category:** Helpfulness

**Method:** LLM-as-Judge with 1-5 score

**Eval prompt:**
```
You are evaluating whether an investment insight is actionable
for a retail investor with basic financial literacy.

[BEGIN DATA]
************
[Company]: {ticker}
[Insight]: {insight_text}
[Verdict]: {verdict}
[Score]: {score}
[END DATA]

Rate this insight on a scale of 1-5:

5 = Highly actionable: References specific numbers, explains WHY
    it matters, and connects to the investment decision.
    Example: "ROE of 28% has been consistent for 5 years,
    indicating a durable competitive moat worth paying a premium for."

3 = Somewhat actionable: States a fact but doesn't explain
    the implication or connect to the decision.
    Example: "The company has strong profit margins."

1 = Not actionable: Generic statement that could apply to any company.
    Example: "This is an interesting investment opportunity."

Your response must be a single number (1-5), followed by a brief explanation.
```

**Rails:** `1` | `2` | `3` | `4` | `5`

---

## Evaluation Layers

### Layer 1: Data Pipeline (Code-Based)
Deterministic checks — no LLM needed. Catches bad data before it reaches agents.

| Check | What it verifies |
|-------|-----------------|
| EPS sanity | `eps_actual` and `eps_estimate` are valid floats, not NaN |
| Beat/miss label | If `actual > estimate`, label is `beat` |
| Surprise math | `surprise_pct` matches `(actual - estimate) / estimate * 100` |
| Price history | No NaN in OHLCV, dates chronological, close > 0 |
| Financial metrics | ROE, P/E, debt-to-equity within plausible ranges |
| JSON safety | All outputs serialize to JSON without errors |

### Layer 2: Agent Output (Code + LLM-as-Judge)
Checks both structure and quality of what agents produce.

**Code-based:**
- Score is between -100 and +100, rounded to 2 decimal places
- Verdict matches score (e.g., score >= 60 → strong_buy)
- Output contains all required keys: score, verdict, insights, concerns
- Same ticker analyzed twice → scores within ±5 points

**LLM-as-Judge:**
- Financial Grounding (Dimension 1)
- Philosophy Adherence (Dimension 2)
- Actionable Quality (Dimension 3)

### Layer 3: Consensus (Code + LLM-as-Judge)
Checks the synthesis step that combines all 5 agents.

- Agreement score is mathematically valid (0-1 range, reflects variance)
- If all agents say "sell", consensus is not "strong_buy"
- Consensus points are coherent given agent results (LLM judge)

---

## Golden Dataset

5 well-known tickers with expected score ranges per agent. These ranges are established from baseline behavior and verified manually.

If a code change pushes a score outside its expected range, the eval catches the regression.

**Why ranges instead of exact scores?** Financial data changes daily. Apple's P/E today is different from next week. But Buffett should *always* score Apple somewhere in the positive range (strong brand moat, high ROE). The range captures the "this should always be roughly positive" invariant.

---

## How to Run

```bash
# All evals
python -m backend.evals.run_evals

# Just data pipeline checks
python -m backend.evals.run_evals --layer 1

# Just agent evals
python -m backend.evals.run_evals --layer 2

# Just consensus evals
python -m backend.evals.run_evals --layer 3
```

LLM-as-Judge evals require `OPENAI_API_KEY`. If not set, only code-based evals run.

---

## What We Intentionally Left Out

- **Toxicity/Safety** — Agents discuss financial metrics, not user-generated content. Low risk.
- **Fluency** — Most insights are rule-based templates. Fluency is already controlled.
- **Arize/Phoenix integration** — Build evals first, add observability later.
- **CI/CD automation** — Run manually until we trust the framework, then automate.

---

*Framework inspired by [Arize LLM Evaluation Guide](https://arize.com/llm-evaluation/metrics/) and Aman Khan's AI Product Sense course (Lesson 4).*
