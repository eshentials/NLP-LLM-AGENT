# Research Paper Analysis Agent

A **multi-step LLM agent** built with the Groq API that takes any research topic and produces a structured, evidence-based research brief through a six-step pipeline.

---

## What the Agent Does

The agent accepts a free-text research topic and chains six steps — five LLM calls and one external tool call — where every step's output becomes structured input for the next.

```
User query
   │
   ▼
Step 1 (LLM)  — Parse query into structured JSON (topic, sub-questions, search query)
   │
   ▼
Step 2 (TOOL) — Web search via Serper API → list of real results {title, snippet, link}
   │
   ▼
Step 3 (LLM)  — Extract key claims, findings, consensus view from search results
   │
   ▼
Step 4 (LLM)  — Critically evaluate evidence quality and methodology
   │
   ▼
Step 5 (LLM)  — Identify research gaps and open questions
   │
   ▼
Step 6 (LLM)  — Synthesize all previous outputs into a structured research brief
   │
   ▼
Outputs: brief_<topic>_<ts>.md  +  state_<topic>_<ts>.json
```

No step can be removed without breaking the chain. Step 3 cannot run without Steps 1 and 2. Step 4 cannot run without Step 3. Steps 5 and 6 depend on all prior outputs.

---

## Chain Design

| Step | Type | Input | Output |
|------|------|-------|--------|
| 1 | LLM | Raw user query | JSON: core_topic, search_query, sub_questions, field, scope |
| 2 | **Tool** (Serper) | search_query from Step 1 | List of {title, snippet, link} |
| 3 | LLM | search results + sub_questions from Steps 1 & 2 | JSON: main_findings, consensus_view, contested_points |
| 4 | LLM | findings + contested_points from Step 3 | JSON: evidence_quality, reliability_score, strengths, weaknesses |
| 5 | LLM | findings from Step 3 + weaknesses from Step 4 | JSON: open_questions, underexplored_angles, practical_implications |
| 6 | LLM | ALL prior outputs combined | Markdown research brief (6 sections) |

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Credentials

Create a `.env` file in this directory before running — required variable names are listed only in `.gitignore` comments (those values must never be committed).

---

## Running the Agent

### Option A — Interactive prompt
```bash
python agent.py
# You will be asked: Enter your research topic:
```

### Option B — Pass topic as argument
```bash
python agent.py effects of microplastics on human gut microbiome
python agent.py transformer attention mechanisms in low-resource NLP
python agent.py effectiveness of cognitive behavioural therapy for insomnia
```

### Option C — Import in your own script
```python
from agent import run_agent
state = run_agent("your topic here", output_dir="my_outputs")
print(state["final_brief"])
```

---

## Outputs

Two files are written to the `outputs/` directory after each run:

| File | Contents |
|------|----------|
| `brief_<topic>_<ts>.md` | Human-readable Markdown research brief with sources |
| `state_<topic>_<ts>.json` | Full pipeline state — every step's raw output, errors, timestamp |

---

## Error Handling

- **Tool failure (Step 2):** if the Serper API returns no results (network error, quota, etc.), the agent logs the error to `state["errors"]` and continues with a stub result. The chain does not crash.
- **JSON parse failures (Steps 1, 3, 4, 5):** if the LLM returns malformed JSON, the agent falls back to a safe default structure and logs the failure. Raw text is preserved.
- All errors are visible in the JSON state file and in the Markdown brief's footer.

---

## Where the Chain Can Break

1. **Hallucination in Step 1:** if the query is very vague, the parsed `search_query` may miss the user's intent, degrading all downstream steps.
2. **Thin search results (Step 2):** for very niche or recent topics, Serper may return few high-quality snippets, making Steps 3–6 work with poor evidence.
3. **Compounding errors:** if Step 3 misidentifies a finding, that error propagates through Steps 4, 5, and 6 without correction.
4. **Length limits:** very broad topics may overflow the LLM context in Step 6, which receives all prior outputs concatenated.

---

## Dependencies

- `groq` — official Groq Python SDK
- `requests` — HTTP calls to Serper API

No LangChain, LlamaIndex, or agent frameworks are used. The chain is built manually.
# NLP-LLM-AGENT
