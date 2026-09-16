# app/travel_system/agents/date_budget_agent.py

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from ..tools.date_tools import calculate_travel_dates
from ..tools.budget_tools import get_exchange_rates_and_budget
from app.core.llm import get_llm
import logging 
from pathlib import Path
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from configs.agent_config import DATE_BUDGET_AGENT_SYSTEM_MESSAGE

def create_date_budget_agent() -> AgentExecutor:
    logging.debug("Creating Date Budget Agent...")
    
    date_budget_prompt = ChatPromptTemplate.from_messages([
        ("system", DATE_BUDGET_AGENT_SYSTEM_MESSAGE),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    date_budget_tools = [calculate_travel_dates, get_exchange_rates_and_budget]
    
    llm_instance = get_llm(temperature=0.0) 
    if not llm_instance:
         logging.error("LLM could not be created for Date Budget Agent!")
         raise ValueError("LLM could not be initialized for Date Budget Agent.")
         
    date_budget_agent = create_tool_calling_agent(
        llm=llm_instance,
        tools=date_budget_tools,
        prompt=date_budget_prompt
    )
    
    date_budget_executor = AgentExecutor.from_agent_and_tools(
        agent=date_budget_agent,
        tools=date_budget_tools,
        verbose=True,
        handle_parsing_errors=True 
    )
    
    logging.debug("Date Budget Agent successfully created.")
    return date_budget_executor
