# app/travel_system/agents/coordinator_agent.py

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from app.core.llm import get_llm
import logging
from pathlib import Path
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from configs.agent_config import TRAVEL_COORDINATOR_SYSTEM_MESSAGE

def create_coordinator_agent() -> AgentExecutor:
    logging.debug("Creating Coordinator Agent...")

    coordinator_prompt = ChatPromptTemplate.from_messages([
        ("system", TRAVEL_COORDINATOR_SYSTEM_MESSAGE),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    coordinator_tools = []

    llm_instance = get_llm(temperature=0.1)
    if not llm_instance:
         logging.error("LLM could not be created for Coordinator Agent!")
         raise ValueError("LLM could not be initialized for Coordinator Agent.")

    coordinator_agent = create_tool_calling_agent(
        llm=llm_instance,
        tools=coordinator_tools,
        prompt=coordinator_prompt
    )

    coordinator_executor = AgentExecutor.from_agent_and_tools(
        agent=coordinator_agent,
        tools=coordinator_tools,
        verbose=True,
        handle_parsing_errors=True
    )

    logging.debug("Coordinator Agent created successfully (stateless).")
    return coordinator_executor
