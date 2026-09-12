---
name: add-agent
description: Scaffold a new specialized leaf agent and wire it into the supervisor graph. Use when the user wants to add a new agent / category / route to the multi-agent chatbot (e.g. "add a weather agent", "add a new agent that does X", "route legal questions to their own agent"). Handles the coordinated edits across app/agents/, configs/app_config.py, configs/agent_config.py, and app/graph.py that are easy to get partially wrong.
---

# Add a new agent to the supervisor graph

Adding an agent touches **5 files** that must stay consistent. Miss one and the graph
either fails to compile or silently never routes to the new agent. Work through the
checklist in order and do not skip the verification step.

## Step 0 — gather inputs from the user

Ask (or infer from the request) and confirm before editing:

| Input | Example | Used for |
|---|---|---|
| **Category name** | `Weather` | the classifier's output label, stored in `state["classification"]` |
| **Node name** | `weather_agent` | LangGraph node id + module filename |
| **Handler fn name** | `handle_weather_query` | the node callable |
| **Agent style** | RAG over a Chroma collection / ReAct with tools / static response / nested sub-graph | which template to start from (Step 4) |
| **One-line scope** | "current & forecast weather for a city" | the classifier prompt section + few-shot example |

Keep the category name short and distinct from the existing five
(`Resmi Gazete`, `News`, `Travel`, `Document Question`, `Other`). Keep everything
user-facing in **English** (see CLAUDE.md).

Below, `<Category>`, `<node>`, `<handler>`, `<CONST>` are placeholders — `<CONST>` is
the UPPER_SNAKE form of the node name (e.g. `NODE_WEATHER`).

## Step 1 — `configs/app_config.py`: add the name constants

In the `# Config for app/graph.py` block, next to `NODE_TRAVEL` / `NODE_AGENTIC_RAG`:

```python
NODE_<CONST> = "<node>"
<CATEGORY_CONST> = "<Category>"   # e.g. WEATHER_CATEGORY = "Weather"
```

Rule from CLAUDE.md: never hardcode these strings elsewhere — `graph.py` keys its node
registration and routing dict off these constants. (The existing router still hardcodes
`"Resmi Gazete"`, `"News"`, `"Travel"` as string literals — do **not** copy that; use your
new constant.)

## Step 2 — `configs/agent_config.py`: register the category with the classifier

Three edits in the `# Config for app/agents/supervisor.py` section:

1. **`VALID_TARGET_CATEGORIES`** — insert `"<Category>"` *before* `"Other"`:
   ```python
   VALID_TARGET_CATEGORIES: List[str] = ["Resmi Gazete", "News", "Travel", "Document Question", "<Category>", "Other"]
   ```

2. **`CLASSIFICATION_PROMPT_TEMPLATE`** — this prompt hardcodes the count and the list:
   - Change "one of the following **five** categories" → "**six**".
   - Add a numbered section for `<Category>` (renumber `Other` last) with a one-line
     definition and 2–3 example questions, matching the style of the existing entries.
   - Add a few-shot line in the "Here are some classification examples:" block:
     ```
     Query: "<a representative query>"
     Category: <Category>
     ```
   - In the final instruction line, update `**five**` → `**six**` and add `'<Category>'`
     to the explicit name list.

3. Add any prompt template your agent needs (e.g. a `WEATHER_PROMPT_TEMPLATE`), keeping
   it in this file with the other templates.

## Step 3 — `app/agents/<node>.py`: create the leaf node

Every leaf node is a `def <handler>(state: Dict[str, Any]) -> Dict[str, Any]` that reads
`state.get("query")` and returns a dict merged into `AgentState` — at minimum
`{"answer": ..., "source": ...}` (add `"context": ...` for RAG). It must **never raise**:
catch exceptions and return an error answer, like the existing agents.

Conventions to copy from siblings:
- Path bootstrap: `project_root = Path(__file__).resolve().parents[2]; sys.path.append(str(project_root))` before importing `app.*` / `configs`.
- LLM: only via `from app.core.llm import get_llm`. `get_llm()` returns **`None`** if
  `GEMINI_API_KEY` is missing — check for it (`llm = get_llm(); if llm is None: return {"answer": "...", "source": "... (Error: LLM unavailable)"}`).
- `logging.basicConfig(...)` line matching the other modules.
- Handle missing/empty `query` up front with an error answer.

### Template A — RAG over a Chroma collection (model after `resmi_gazete_agent.py`)

```python
# app/agents/<node>.py
import logging, sys
from pathlib import Path
from typing import Dict, Any, Optional

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from app.tools.rag_tools import retrieve_documents, format_context
from app.core.llm import get_llm
from configs.agent_config import <COLLECTION_CONST>, NUM_DOCUMENTS_TO_RETRIEVE, <PROMPT_CONST>

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')

def <handler>(state: Dict[str, Any]) -> Dict[str, Any]:
    query: Optional[str] = state.get("query")
    source_info = "<Category> (Collection: {})".format(<COLLECTION_CONST>)
    if not query:
        return {"answer": "I received an incomplete query. Please rephrase.", "context": None, "source": source_info + " (Error: Missing Query)"}
    try:
        docs = retrieve_documents(query=query, collection_name=<COLLECTION_CONST>, n_results=NUM_DOCUMENTS_TO_RETRIEVE)
    except Exception as e:
        logging.error(f"Retrieval error: {e}", exc_info=True)
        return {"answer": "An issue occurred while accessing the knowledge base.", "context": None, "source": source_info + " (Error: Retrieval)"}
    if not docs:
        return {"answer": f"I couldn't find anything related to '{query}'.", "context": None, "source": source_info + " (No Results)"}
    context = format_context(docs)
    llm = get_llm()
    if llm is None:
        return {"answer": "The language model is unavailable right now.", "context": context, "source": source_info + " (Error: LLM unavailable)"}
    try:
        resp = llm.invoke(<PROMPT_CONST>.format(query=query, context=context))
        return {"answer": resp.content.strip(), "context": context, "source": source_info + " (Generated via RAG)"}
    except Exception as e:
        logging.error(f"LLM error: {e}", exc_info=True)
        return {"answer": "Information was found, but synthesizing the answer failed.", "context": context, "source": source_info + " (Error: LLM)"}
```

New Chroma collection notes (see CLAUDE.md / `app/storage/database.py`): the pre-built
corpus uses `intfloat/multilingual-e5-large`; ad-hoc uploaded-doc RAG uses
`models/embedding-001`. Pick one per collection and don't mix. You will also need a
pipeline/step to populate the collection.

### Template B — ReAct agent with tools (model after `news_agent.py`)

Cache an `AgentExecutor` at module level, build it with `create_react_agent(get_llm(temperature=0.7), tools, prompt)`
wrapped in `AgentExecutor(..., handle_parsing_errors=True, max_iterations=6)`, and in
`<handler>` call `agent_executor.invoke({"input": augmented_query})` where
`augmented_query` appends `"(Answer in English...)"`. Read the answer from
`response.get("output")`. Put tools in `app/tools/` and import them.

### Template C — static / deterministic (model after `fallback_agent.py`)

Just return `{"answer": <CONST_RESPONSE>, "source": "<Category> Agent"}` with the string
defined in `configs/agent_config.py`.

### Template D — nested sub-graph (model after `travel_agent.py`)

Thin wrapper that invokes a separate `StateGraph` built in its own module (see
`app/travel_system/`). Only do this if the user explicitly wants an isolated multi-step
pipeline.

## Step 4 — `app/graph.py`: register and route

Four edits (note: the file's top comment header is stale — it's really `app/graph.py`):

1. **Import the handler** alongside the others:
   ```python
   from app.agents.<node> import <handler>
   ```
2. **Import the constants** — add `NODE_<CONST>` and `<CATEGORY_CONST>` to the
   `from configs.app_config import (...)` list.
3. **`route_based_on_classification`** — add a branch before the `else`:
   ```python
   elif classification_result == <CATEGORY_CONST>:
       return NODE_<CONST>
   ```
4. **Graph wiring** — three lines mirroring every other leaf:
   ```python
   workflow.add_node(NODE_<CONST>, <handler>)
   # ... in the add_conditional_edges routing dict:
   NODE_<CONST>: NODE_<CONST>,
   # ... after the other add_edge calls:
   workflow.add_edge(NODE_<CONST>, END)
   ```
   The routing dict keys must cover **every** value the router can return. (The dict
   also has a legacy `BELGE_SORUSU_CATEGORY: NODE_AGENTIC_RAG` entry that the router
   never actually returns — harmless; you don't need an analogous one.)

## Step 5 — docs

`CLAUDE.md` enumerates the leaf nodes under "Main graph" and "Leaf nodes" — add the new
agent there. Update `README.md` if it lists the agents. (The repo's PostToolUse hook will
also remind you.)

## Step 6 — verify

No test suite exists, so check by hand:

```bash
source .venv/bin/activate
# 1. Graph compiles and the node is registered:
python -c "from app.graph import graph_app; ns=graph_app.get_graph().nodes; assert '<node>' in ns, ns; print('node OK:', sorted(ns))"
# 2. Constants line up:
python -c "from configs.app_config import NODE_<CONST>, <CATEGORY_CONST>; from configs.agent_config import VALID_TARGET_CATEGORIES as v; assert <CATEGORY_CONST> in v, v; print('config OK')"
# 3. Classifier routes a representative query (needs GEMINI_API_KEY; costs a quota call):
python -c "from app.agents.supervisor import classify_query; print(classify_query({'query': '<representative query>'}))"
# 4. End-to-end through the graph:
python -c "from app.graph import graph_app; print(graph_app.invoke({'query': '<representative query>'})['source'])"
```

Then run the app (`streamlit run app/ui/streamlit_app.py --server.fileWatcherType none`)
and try a query that should hit the new agent plus one that should NOT (to check you
didn't cannibalise another category's routing).

## Gotchas

- **Classifier is an LLM** — adding the category to `VALID_TARGET_CATEGORIES` is not
  enough; the prompt section + few-shot example are what actually make Gemini emit the
  label. Test with real queries.
- **`state["classification"]` holds the category string, not the node name.** The router
  translates category → `NODE_*`. Keep those two layers straight.
- **Every router return value needs a key in the `add_conditional_edges` dict** or
  LangGraph raises at compile time.
- **`get_llm()` can return `None`.** Guard it.
- **Leaf nodes must not raise** — classification and routing must never crash the graph.
- **Gemini free tier is quota-limited** (~5 req/min on some models). The verify steps
  above make LLM calls; space them out if you hit 429s.
- **English only** for prompts and user-facing text, even though source data is Turkish.
