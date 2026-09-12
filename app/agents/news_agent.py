# app/agents/news_agent.py

import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain_core.prompts import PromptTemplate

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from configs.agent_config import LANGCHAIN_HUB_AVAILABLE, REACT_HUB_PROMPT_PATH, MANUAL_REACT_PROMPT_TEMPLATE

from app.tools.external_apis import wikipedia_tool, web_search_tool
from app.core.llm import get_llm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')

NEWS_AGENT_TOOLS = [wikipedia_tool, web_search_tool]

# We cache the AgentExecutor instead of creating it every time
agent_executor: Optional[AgentExecutor] = None

# Function used to get or create the AgentExecutor
def get_news_agent_executor() -> Optional[AgentExecutor]:
    global agent_executor
    if agent_executor:
        logging.debug("Returning News Agent Executor from cache.")
        return agent_executor

    logging.info("Creating new News Agent Executor...")
    try:
        llm = get_llm(temperature=0.7)
        tools = NEWS_AGENT_TOOLS
        prompt = None

        # If Langchain Hub is available, try fetching the ReAct prompt from there.
        # If not, fall back to the manually created prompt.
        if LANGCHAIN_HUB_AVAILABLE:
            try:
                prompt = hub.pull(REACT_HUB_PROMPT_PATH)
                logging.info(f"Prompt fetched from Langchain Hub: {REACT_HUB_PROMPT_PATH}")
            except Exception as e:
                logging.warning(f"Failed to fetch prompt from Langchain Hub ({e}). Using manual prompt instead.")
                prompt = None

        if prompt is None:
            tool_descriptions = "\n".join([f"{t.name}: {t.description}" for t in tools])
            tool_names = ", ".join([t.name for t in tools])
            prompt = PromptTemplate.from_template(MANUAL_REACT_PROMPT_TEMPLATE).partial(
                tools=tool_descriptions,
                tool_names=tool_names
            )
            logging.info("Using manually defined ReAct prompt.")

        # Create the news agent
        agent = create_react_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=6
        )
        logging.info("News Agent Executor successfully created.")
        return agent_executor

    except Exception as e:
        logging.error(f"Error while creating News Agent Executor: {e}", exc_info=True)
        return None

# Function used to run the agent
def handle_news_query(state: Dict[str, Any]) -> Dict[str, Any]:
    logging.info("Running News Agent...")
    query: Optional[str] = state.get("query")  # Query is fetched from the state
    final_answer: str = "An unexpected error occurred while processing your news or general information query."
    source_info = "News Agent (Web/Wikipedia)"  # Default source information for the response

    # If there's no query or it's empty, return an error message
    if not query:
        logging.error("News Agent: No valid 'query' found in the state.")
        final_answer = "I received an unrecognized or incomplete query."
        source_info += " (Error: Missing Query)"
        return {"answer": final_answer, "source": source_info}

    logging.info(f"Query to be processed: '{query}'")

    agent_executor = get_news_agent_executor()

    logging.info("Running News Agent Executor...")

    augmented_query = f"{query}\n\n(Answer in English, regardless of the language of the question or any retrieved source content.)"
    response = agent_executor.invoke({"input": augmented_query})

    # The agent's final answer is found under the 'output' key
    final_answer = response.get("output")
    if not final_answer:
        logging.warning("Agent Executor did not produce an 'output'.")
        final_answer = "Your request was processed but no answer was generated. Please try rephrasing your question."
        source_info += " (Agent Did Not Respond)"
    else:
        logging.info("News Agent Executor completed successfully.")
        source_info += " (Agent Successful)"

    logging.info(f"News Agent completed. Response (first 100 characters): '{final_answer[:100]}...'")
    return {"answer": final_answer, "source": source_info}
