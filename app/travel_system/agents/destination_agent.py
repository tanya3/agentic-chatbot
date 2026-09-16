# app/travel_system/agents/destination_agent.py

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from ..tools.destination_tools import (
    search_city_info, 
    get_weather_forecast, 
    search_hotel_booking_links, 
    get_tomtom_map_url
)
from app.core.llm import get_llm
import logging
from pathlib import Path
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from configs.agent_config import DESTINATION_RESEARCH_AGENT_SYSTEM_MESSAGE

def create_destination_agent() -> AgentExecutor:
    logging.debug("Creating Destination Agent...")
    
    destination_prompt = ChatPromptTemplate.from_messages([
        ("system", DESTINATION_RESEARCH_AGENT_SYSTEM_MESSAGE),
        ("human", "{input}"), 
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    destination_tools = [
        search_city_info, 
        get_weather_forecast, 
        search_hotel_booking_links,
        get_tomtom_map_url
    ]
    logging.debug(f"Tools for Destination Agent: {[tool.name for tool in destination_tools]}")

    llm_instance = get_llm() 
    if not llm_instance:
        logging.error("LLM could not be created for Destination Agent!")
        raise ValueError("LLM could not be initialized for Destination Agent.")

    destination_agent_runnable = create_tool_calling_agent(
        llm=llm_instance,
        tools=destination_tools,
        prompt=destination_prompt
    )

    destination_executor = AgentExecutor.from_agent_and_tools(
        agent=destination_agent_runnable,
        tools=destination_tools,
        verbose=True, 
        handle_parsing_errors=True, 
    )
    
    logging.debug("Destination Agent successfully created.")
    return destination_executor
