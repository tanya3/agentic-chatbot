# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A **supervisor-based multi-agent chatbot** built with LangChain/LangGraph, served via Streamlit. A supervisor node classifies each user query and routes it to one specialized agent (Resmi Gazete RAG, News, Travel, Agentic RAG over an uploaded doc, or Fallback). All LLM calls go through Google Gemini. The underlying source data is Turkish (it RAGs over the Turkish "Resmi Gazete" official gazette and Turkish news sources), but the UI and all agent prompts instruct **English** output — agents translate/synthesize from the Turkish context into English answers rather than passing it through. Keep new prompts and user-facing text in English unless told otherwise.

## Commands

There is no test suite, linter, or build step configured in this repo (no `pytest`/`tox`/`ruff`/`Makefile`/`pyproject.toml`). Development is run/verify-by-hand via Streamlit.

Setup (Python 3.11 recommended; 3.10+ required — the LangChain/LangGraph stack in
`requirements.txt` is pinned to the 0.3.x line, and the planned weather MCP server needs
3.10+):
```bash
python3.11 -m venv .venv
source .venv/bin/activate      # .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Run the app locally:
```bash
streamlit run app/ui/streamlit_app.py --server.fileWatcherType none
# -> http://localhost:8501
```

Run with Docker:
```bash
docker compose build
docker compose up
```

Data pipeline (only needed if `data/embeddings/` is empty or you want to refresh the corpus — run in order):
```bash
python scripts/del.py                  # optional: wipe data/ dir
python scripts/news_fetcher.py         # fetch TRT Haber news -> data/raw/haberler
python scripts/resmi_news_fetcher.py   # fetch AA Resmi Gazete news -> data/raw/resmi_gazete
python scripts/process_data.py         # chunk raw data -> data/processed/*.jsonl
python scripts/generate_embeddings.py  # embed processed data into ChromaDB (data/embeddings/)
```

Required `.env` keys (loaded via `configs/api_config.py`, dotenv): `GEMINI_API_KEY` (hard-required, everything breaks without it), plus `TAVILY_API_KEY`, `SERPER_API_KEY`, `OPENWEATHERMAP_API_KEY`, `EXCHANGERATE_API_KEY`, and a TomTom key for map URLs used by the travel system.

## Architecture

### Main graph (`app/graph.py`)

A single `StateGraph[AgentState]` (state schema in `app/core/state.py`) with one entry point and five terminal nodes (each routes straight to `END` — no multi-turn loop at this level):

1. `NODE_SUPERVISOR` → `app/agents/supervisor.py::classify_query` — calls Gemini at `temperature=0.0` to classify the query into one of `VALID_TARGET_CATEGORIES` (`configs/agent_config.py`). Falls back to `DEFAULT_TARGET_CATEGORY` on any LLM error or unparseable output — never let classification raise.
2. `route_based_on_classification` (in `app/agents/agentic_rag_agent.py`, despite the name) dispatches on `state["classification"]`, with a `route_directly_to_agentic_rag` flag in state that bypasses classification entirely (used when the UI already knows the user uploaded a document).
3. Leaf nodes, each `add_edge(..., END)`:
   - `resmi_gazete_agent.py` — RAG over the pre-built Resmi Gazete ChromaDB collection
   - `news_agent.py` — ReAct agent (Tavily/DuckDuckGo + Wikipedia tools)
   - `travel_agent.py` — thin wrapper that invokes the separate `TravelPlanningSystem` sub-graph
   - `agentic_rag_agent.py` — RAG over a user-uploaded document (own ChromaDB collection, different embedding model)
   - `fallback_agent.py` — static `FALLBACK_RESPONSE`

Node name strings and category constants (`NODE_SUPERVISOR`, `NODE_RESMI_GAZETE`, `BELGE_SORUSU_CATEGORY`, etc.) live in `configs/app_config.py` — always use these constants rather than hardcoding the strings, since `graph.py`'s node registration and routing dict both key off them.

### Shared core

- `app/core/llm.py::get_llm()` — the only way agents should construct an LLM. Caches `ChatGoogleGenerativeAI` instances by `(model_name, temperature, max_output_tokens, top_p, top_k, kwargs)` tuple; returns `None` (not an exception) if `GEMINI_API_KEY` is missing, so callers must check for `None`. Default `model_name` is `"gemini-flash-latest"` (a rolling alias Google keeps pointed at a current model) — deliberately not a pinned dated model, since pinned Gemini model names get deprecated/retired over time and start 404ing. Note the free tier is quota-limited per model (as low as 5 requests/minute and low daily caps on some models), which is easy to exhaust while testing multiple agents in a session.
- `app/core/state.py::AgentState` — the `TypedDict` threaded through every top-level graph node (`query`, `classification`, `context`, `answer`, `source`, `pdf_path`, `uploaded_file_data`/`uploaded_file_name`, `route_directly_to_agentic_rag`).
- `app/storage/database.py` — ChromaDB access layer, module-level cached `client`/`embedding_function` singletons. `get_or_create_collection()` always sets `hnsw:space: cosine`. Two different embedding models are used depending on collection: `intfloat/multilingual-e5-large` (default, `MODEL_NAME` in `configs/app_config.py`) for the pre-built Resmi Gazete/news corpus, vs. `models/embedding-001` for ad hoc uploaded-doc RAG — don't mix them across a collection.

### Travel subsystem (`app/travel_system/`)

Architecturally isolated nested LangGraph workflow, invoked as a single black-box step from the main graph via `travel_agent.py`. `workflow.py::TravelPlanningSystem` builds its own `StateGraph[TravelPlanState]` (separate from `AgentState`) with a `MemorySaver` checkpointer and a linear-ish pipeline: `parse_request` → `calculate_dates` → `process_date_budget` → `process_destination` → `compile_final_plan`, with conditional short-circuits to `compile_final_plan` if parsing or date calculation fails. Each of the three "agent" nodes (`coordinator_agent`, `date_budget_agent`, `destination_agent` in `travel_system/agents/`) is itself a small ReAct-style LangChain agent invoked synchronously with a big inline prompt that hardcodes the expected English section headings (e.g. `City Information`, `Weather/Clothing Recommendations`, `Trip Summary`) — if you change the output structure, update the prompt strings in both `configs/agent_config.py` (the agents' system messages) and `workflow.py` (the per-node task prompts), keeping the heading names in sync across both. Tools live in `travel_system/tools/` (budget/currency, date math, destination lookups via weather/hotel/TomTom-maps APIs, query parsing); PDF export is `travel_system/utils/pdf_saver.py` using the DejaVu fonts in `assets/fonts/`.

### Config layout (`configs/`)

- `app_config.py` — node/category name constants, Chroma path + embedding model name, document loader mapping, travel city→currency map. Also does the `sys.path` bootstrap most modules copy.
- `agent_config.py` — all prompt templates and classification categories (`VALID_TARGET_CATEGORIES`, `CLASSIFICATION_PROMPT_TEMPLATE`, ReAct prompt, fallback response text).
- `api_config.py` — `load_env()` loads `.env` via `python-dotenv`; imported (and thus executed) for its side effect wherever API keys are needed.
- `script_config.py` — settings for the offline `scripts/` data pipeline.

### Path bootstrap pattern

Most modules manually do `project_root = Path(__file__).resolve().parents[N]; sys.path.append(str(project_root))` before importing sibling packages (`app.core`, `configs`, etc.) rather than relying on package installation — there's no `pyproject.toml`/`setup.py`, so this `sys.path` hack is how cross-package imports (e.g. `configs` from `app/`) resolve. Follow the same pattern in any new module that needs to import across `app/`/`configs`, matching the existing relative depth (`parents[N]`) for that file's location.

