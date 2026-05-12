"""
Research Paper Analysis Agent
A multi-step LLM agent that analyzes a research topic through 6 chained steps:
1. LLM - Parse & structure the user's research query
2. Tool - Web search for recent papers/articles on the topic
3. LLM - Extract key claims and findings from search results
4. LLM - Critically evaluate methodology and evidence quality
5. LLM - Identify research gaps and open questions
6. LLM - Synthesize everything into a structured research brief
"""

import json
import os
from pathlib import Path

import requests
from groq import Groq
from datetime import datetime

# ─── Configuration (credentials only from environment / `.env`; never committed) ─
MODEL = "llama-3.3-70b-versatile"


def _load_dotenv_local() -> None:
    """Populate os.environ from a project-root `.env` if present."""
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, rest = line.partition("=")
        key = key.strip()
        val = rest.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = val


def _need_env(name: str) -> str:
    raw = os.environ.get(name)
    val = raw.strip() if raw else ""
    if not val:
        raise RuntimeError(
            f"Missing required env var '{name}'. Add it to a `.env` file in the project root "
            f"(see key names listed in `.gitignore` comments)."
        )
    return val


_load_dotenv_local()
GROQ_API_KEY = _need_env("GROQ_API_KEY")
SERPER_API_KEY = _need_env("SERPER_API_KEY")

client = Groq(api_key=GROQ_API_KEY)


# ─── Shared State ──────────────────────────────────────────────────────────────
def init_state(user_query: str) -> dict:
    return {
        "user_query":       user_query,
        "parsed_query":     None,   # Step 1 output
        "search_results":   None,   # Step 2 output (tool)
        "extracted_claims": None,   # Step 3 output
        "critique":         None,   # Step 4 output
        "research_gaps":    None,   # Step 5 output
        "final_brief":      None,   # Step 6 output
        "errors":           [],
        "timestamp":        datetime.now().isoformat(),
    }


# ─── LLM Helper ────────────────────────────────────────────────────────────────
def call_llm(system_prompt: str, user_prompt: str, step_name: str) -> str:
    print(f"\n{'─'*50}")
    print(f"  🤖  LLM  |  {step_name}")
    print(f"{'─'*50}")
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=1500,
    )
    result = response.choices[0].message.content.strip()
    print(f"  ✓  Done ({len(result)} chars)")
    return result


# ─── Tool: Web Search ──────────────────────────────────────────────────────────
def tool_web_search(query: str, num_results: int = 8) -> list[dict]:
    """
    Calls Serper API to retrieve real web/news search results.
    Returns a list of {title, snippet, link} dicts.
    """
    print(f"\n{'─'*50}")
    print(f"  🔍  TOOL  |  Web Search")
    print(f"  Query: {query}")
    print(f"{'─'*50}")

    url     = "https://google.serper.dev/search"
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    payload = {"q": query, "num": num_results, "gl": "us", "hl": "en"}

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=10)
        resp.raise_for_status()
        data    = resp.json()
        results = []

        for item in data.get("organic", [])[:num_results]:
            results.append({
                "title":   item.get("title", ""),
                "snippet": item.get("snippet", ""),
                "link":    item.get("link", ""),
            })

        print(f"  ✓  Retrieved {len(results)} results")
        return results

    except requests.RequestException as e:
        print(f"  ✗  Search failed: {e}")
        return []   # graceful fallback — empty list, chain continues


# ─── Step 1: Parse & Structure the Query ───────────────────────────────────────
def step1_parse_query(state: dict) -> dict:
    system = """You are a research analyst. Your job is to parse a user's research topic
into a structured form. Respond ONLY with valid JSON — no markdown fences, no explanation.

Return this exact shape:
{
  "core_topic": "...",
  "search_query": "...",
  "sub_questions": ["...", "...", "..."],
  "field": "...",
  "scope": "narrow | medium | broad"
}"""

    user = f"Parse this research topic: {state['user_query']}"

    raw = call_llm(system, user, "Step 1 — Parse Query")

    try:
        state["parsed_query"] = json.loads(raw)
    except json.JSONDecodeError:
        # fallback: extract search_query heuristically
        state["parsed_query"] = {
            "core_topic":   state["user_query"],
            "search_query": state["user_query"] + f" research {datetime.now().year}",
            "sub_questions": [],
            "field":        "general",
            "scope":        "medium",
        }
        state["errors"].append("Step 1: JSON parse failed, used fallback.")

    return state


# ─── Step 2: Tool — Web Search ─────────────────────────────────────────────────
def step2_web_search(state: dict) -> dict:
    search_q = state["parsed_query"]["search_query"]
    results  = tool_web_search(search_q)

    if not results:
        state["errors"].append("Step 2: No search results returned.")
        # Provide a stub so the chain can still continue
        state["search_results"] = [{"title": "No results", "snippet": "Search unavailable.", "link": ""}]
    else:
        state["search_results"] = results

    return state


# ─── Step 3: Extract Key Claims ─────────────────────────────────────────────────
def step3_extract_claims(state: dict) -> dict:
    snippets = "\n".join(
        f"[{i+1}] {r['title']}: {r['snippet']}"
        for i, r in enumerate(state["search_results"])
    )
    sub_questions = "\n".join(
        f"- {q}" for q in state["parsed_query"].get("sub_questions", [])
    )

    system = """You are a research analyst extracting factual claims from search result snippets.
Respond ONLY with valid JSON — no markdown fences.

Return:
{
  "main_findings": ["finding 1", "finding 2", ...],
  "key_authors_or_sources": ["..."],
  "consensus_view": "...",
  "contested_points": ["..."]
}"""

    user = f"""Topic: {state['parsed_query']['core_topic']}

Sub-questions to address:
{sub_questions}

Search results:
{snippets}

Extract the key claims and findings."""

    raw = call_llm(system, user, "Step 3 — Extract Claims")

    try:
        state["extracted_claims"] = json.loads(raw)
    except json.JSONDecodeError:
        state["extracted_claims"] = {"main_findings": [raw], "key_authors_or_sources": [], "consensus_view": "", "contested_points": []}
        state["errors"].append("Step 3: JSON parse failed, stored raw text.")

    return state


# ─── Step 4: Critical Evaluation ───────────────────────────────────────────────
def step4_critical_evaluation(state: dict) -> dict:
    findings_text = "\n".join(f"- {f}" for f in state["extracted_claims"].get("main_findings", []))
    contested     = "\n".join(f"- {c}" for c in state["extracted_claims"].get("contested_points", []))

    system = """You are a rigorous academic peer reviewer. Critically evaluate the quality
of evidence and methodology behind these research findings.
Respond ONLY with valid JSON — no markdown fences.

Return:
{
  "evidence_quality": "high | medium | low",
  "methodological_concerns": ["..."],
  "reliability_score": 1-10,
  "strengths": ["..."],
  "weaknesses": ["..."],
  "overall_verdict": "..."
}"""

    user = f"""Topic: {state['parsed_query']['core_topic']}
Field: {state['parsed_query']['field']}

Main findings:
{findings_text}

Contested points:
{contested}

Critically evaluate the evidence quality and methodology."""

    raw = call_llm(system, user, "Step 4 — Critical Evaluation")

    try:
        state["critique"] = json.loads(raw)
    except json.JSONDecodeError:
        state["critique"] = {"evidence_quality": "unknown", "overall_verdict": raw, "reliability_score": 5, "strengths": [], "weaknesses": [], "methodological_concerns": []}
        state["errors"].append("Step 4: JSON parse failed, stored raw text.")

    return state


# ─── Step 5: Research Gaps ─────────────────────────────────────────────────────
def step5_research_gaps(state: dict) -> dict:
    findings_text = "\n".join(f"- {f}" for f in state["extracted_claims"].get("main_findings", []))
    weaknesses    = "\n".join(f"- {w}" for w in state["critique"].get("weaknesses", []))

    system = """You are a research strategist identifying unexplored territory in a field.
Respond ONLY with valid JSON — no markdown fences.

Return:
{
  "open_questions": ["..."],
  "underexplored_angles": ["..."],
  "suggested_next_studies": ["..."],
  "practical_implications": ["..."]
}"""

    user = f"""Topic: {state['parsed_query']['core_topic']}

What we know:
{findings_text}

Known weaknesses in current research:
{weaknesses}

Identify research gaps, open questions, and future directions."""

    raw = call_llm(system, user, "Step 5 — Research Gaps")

    try:
        state["research_gaps"] = json.loads(raw)
    except json.JSONDecodeError:
        state["research_gaps"] = {"open_questions": [raw], "underexplored_angles": [], "suggested_next_studies": [], "practical_implications": []}
        state["errors"].append("Step 5: JSON parse failed, stored raw text.")

    return state


# ─── Step 6: Synthesize Final Brief ────────────────────────────────────────────
def step6_synthesize_brief(state: dict) -> dict:
    # Build a rich context from ALL previous steps
    context = f"""
TOPIC: {state['parsed_query']['core_topic']}
FIELD: {state['parsed_query']['field']}

MAIN FINDINGS:
{chr(10).join('- ' + f for f in state['extracted_claims'].get('main_findings', []))}

CONSENSUS VIEW:
{state['extracted_claims'].get('consensus_view', 'N/A')}

EVIDENCE QUALITY: {state['critique'].get('evidence_quality', 'N/A')} (Score: {state['critique'].get('reliability_score', 'N/A')}/10)

STRENGTHS:
{chr(10).join('- ' + s for s in state['critique'].get('strengths', []))}

WEAKNESSES:
{chr(10).join('- ' + w for w in state['critique'].get('weaknesses', []))}

OVERALL VERDICT: {state['critique'].get('overall_verdict', 'N/A')}

OPEN QUESTIONS:
{chr(10).join('- ' + q for q in state['research_gaps'].get('open_questions', []))}

PRACTICAL IMPLICATIONS:
{chr(10).join('- ' + p for p in state['research_gaps'].get('practical_implications', []))}
"""

    system = """You are a senior research director writing an executive research brief.
Write in clear, authoritative prose. Structure your response with these EXACT markdown headers:

## Executive Summary
## Key Findings
## Evidence Assessment
## Research Gaps
## Practical Implications
## Recommended Next Steps

Be specific, cite the findings, and be honest about uncertainty. 2-3 paragraphs per section."""

    user = f"Write a comprehensive research brief based on this analysis:\n{context}"

    state["final_brief"] = call_llm(system, user, "Step 6 — Synthesize Final Brief")
    return state


# ─── Output Writer ─────────────────────────────────────────────────────────────
def write_output(state: dict, output_dir: str = ".") -> str:
    os.makedirs(output_dir, exist_ok=True)
    safe_topic = "".join(c if c.isalnum() or c in " _-" else "_" for c in state["parsed_query"]["core_topic"])[:40].strip()
    ts         = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Markdown brief (human-readable)
    md_path = os.path.join(output_dir, f"brief_{safe_topic}_{ts}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Research Brief: {state['parsed_query']['core_topic']}\n")
        f.write(f"*Generated: {state['timestamp']}*\n\n")
        f.write(state["final_brief"])
        f.write("\n\n---\n\n## Sources\n")
        for r in state.get("search_results", []):
            if r.get("link"):
                f.write(f"- [{r['title']}]({r['link']})\n")
        if state["errors"]:
            f.write("\n\n## Pipeline Errors\n")
            for e in state["errors"]:
                f.write(f"- {e}\n")

    # 2. Full state JSON (for inspection / debugging)
    json_path = os.path.join(output_dir, f"state_{safe_topic}_{ts}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)

    return md_path, json_path


# ─── Main Pipeline ─────────────────────────────────────────────────────────────
def run_agent(user_query: str, output_dir: str = "outputs") -> dict:
    print(f"\n{'═'*50}")
    print(f"  RESEARCH AGENT — Starting Pipeline")
    print(f"  Query: {user_query}")
    print(f"{'═'*50}")

    state = init_state(user_query)

    # Chain execution — each step mutates shared state
    state = step1_parse_query(state)
    state = step2_web_search(state)
    state = step3_extract_claims(state)
    state = step4_critical_evaluation(state)
    state = step5_research_gaps(state)
    state = step6_synthesize_brief(state)

    # Write outputs
    md_path, json_path = write_output(state, output_dir)

    print(f"\n{'═'*50}")
    print(f"  ✅  Pipeline Complete!")
    print(f"  📄  Brief:  {md_path}")
    print(f"  📦  State:  {json_path}")
    if state["errors"]:
        print(f"  ⚠️   Errors: {len(state['errors'])} (see state JSON)")
    print(f"{'═'*50}\n")

    return state


# ─── CLI Entry Point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("Enter your research topic: ").strip()
        if not query:
            query = "effects of sleep deprivation on cognitive performance in adults"

    run_agent(query)
