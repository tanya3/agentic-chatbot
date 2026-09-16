# app/travel_system/workflow.py
# SYNCHRONOUS VERSION

import json
import logging
from typing import TypedDict, Optional, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.runnables import Runnable
from langgraph.checkpoint.memory import MemorySaver

from .agents.coordinator_agent import create_coordinator_agent
from .agents.date_budget_agent import create_date_budget_agent
from .agents.destination_agent import create_destination_agent
from .tools.date_tools import calculate_travel_dates
from .tools.parsing_tools import parse_travel_query

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')

def extract_agent_output_text(output: Any) -> str:
    """Normalize an AgentExecutor's 'output' value to a plain string.

    ChatAnthropic (unlike the previous Gemini wrapper) can return AIMessage.content
    as a list of content blocks (e.g. [{"type": "text", "text": "..."}]) even for a
    plain text completion once tools are bound via create_tool_calling_agent, instead
    of collapsing a single text block down to a bare string.
    """
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        return "".join(
            block.get("text", "")
            for block in output
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return str(output)

class TravelPlanState(TypedDict):
    user_query: str
    origin: Optional[str]
    parsed_request: Optional[Dict[str, Any]]
    calculated_dates: Optional[Dict[str, str]]
    date_budget_summary: Optional[str]
    destination_summary: Optional[str]
    final_plan: Optional[str]
    error_message: Optional[str]

class TravelPlanningSystem:
    def __init__(self):
        logging.info("Initializing TravelPlanningSystem (Synchronous Version)...")
        self.coordinator_agent = create_coordinator_agent()
        self.date_budget_agent = create_date_budget_agent()
        self.destination_agent = create_destination_agent()
        self.app = self.build_graph()
        logging.info("TravelPlanningSystem successfully initialized (Synchronous Version).")

    def parse_request_node(self, state: TravelPlanState) -> Dict[str, Any]: # def
        logging.info("[ParseNode] Running...")
        user_query = state['user_query']
        try:
            parsed_info = parse_travel_query.func(user_query=user_query)
            logging.info(f"[ParseNode] Parsing Result: {parsed_info}")
            error_message = parsed_info.get("error")
            required_fields = ["destination", "natural_language_date", "duration_days"]
            missing_fields = [field for field in required_fields if not parsed_info.get(field)]
            if missing_fields: error_message = parsed_info.get("error", f"Missing information: {', '.join(missing_fields)}"); logging.warning(f"[ParseNode] Missing: {missing_fields}")
            return {
                "parsed_request": parsed_info,
                "origin": parsed_info.get("origin"),
                "error_message": error_message
            }
        except Exception as e: logging.error(f"[ParseNode] Error: {e}", exc_info=True); return {"error_message": f"Query could not be parsed: {e}"}

    def calculate_dates_node(self, state: TravelPlanState) -> Dict[str, Any]: # def
        logging.info("[CalculateDatesNode] Running...")
        parsed_info = state.get('parsed_request')
        if not parsed_info: logging.warning("[CalculateDatesNode] Information missing."); return {"error_message": "Cannot calculate dates without parsed info."}
        natural_language_date = parsed_info.get("natural_language_date"); duration_days = parsed_info.get("duration_days")
        if not natural_language_date or duration_days is None: logging.warning("[CalculateDatesNode] Date/duration missing."); return {"error_message": "Missing nl_date or duration."}
        try:
            dates_result = calculate_travel_dates.func(natural_language_date=natural_language_date, duration_days=duration_days)
            logging.info(f"[CalculateDatesNode] Result: {dates_result}")
            if "error" in dates_result: logging.error(f"[CalculateDatesNode] Tool error: {dates_result['error']}"); return {"error_message": f"Date calculation failed: {dates_result['error']}"}
            return {"calculated_dates": dates_result, "error_message": None}
        except Exception as e: logging.error(f"[CalculateDatesNode] Error: {e}", exc_info=True); return {"error_message": f"Error calculating dates: {e}"}


    def process_date_budget_node(self, state: TravelPlanState) -> Dict[str, Any]: 
        logging.info("[DateBudgetNode] Running...")
        parsed_info = state.get('parsed_request'); calculated_dates = state.get('calculated_dates')
        if not parsed_info or not calculated_dates: logging.warning("[DateBudgetNode] Information missing."); return {"date_budget_summary": "Skipped: Missing info."}
        destination = parsed_info.get('destination'); nl_date = parsed_info.get('natural_language_date'); start_date = calculated_dates.get('start_date'); end_date = calculated_dates.get('end_date'); duration = parsed_info.get('duration_days'); budget_amount = parsed_info.get('budget_amount', 'N/A'); budget_currency = parsed_info.get('budget_currency', '')
        if not all([destination, nl_date, start_date, end_date, duration is not None]): logging.warning("[DateBudgetNode] Sub-information missing."); return {"date_budget_summary": "Skipped: Missing sub-keys."}

        date_budget_query = f"""Analyze the dates and budget for a trip.
        Destination: {destination}
        Dates: {nl_date} (Calculated as {start_date} to {end_date}, Duration: {duration} days)
        Budget: {budget_amount} {budget_currency}

        Provide a brief summary in ENGLISH covering:
        1. Confirmation of dates and duration.
        2. Budget amount and currency. Mention if currency conversion might be needed (if not TRY).
        3. A very brief note if the budget seems reasonable for the destination/duration (optional, simple check).
        Respond ONLY with the summary. Do not add any extra text.
        """
        logging.info("[DateBudgetNode] Calling Date Budget Agent (SYNCHRONOUS)...")
        try:
            agent_input = {"input": date_budget_query}
            response = self.date_budget_agent.invoke(agent_input)
            logging.info(f"[DateBudgetNode] Agent Raw Response: {response}")
            summary = extract_agent_output_text(response.get("output", "Date/Budget summary error."))
            logging.info(f"[DateBudgetNode] Agent Summary Result: {summary}")
            error_in_summary = "error" in summary.lower() or "hata" in summary.lower()
            return {"date_budget_summary": summary, "error_message": state.get("error_message") or (f"DateBudget Agent Error: {summary}" if error_in_summary else None)}
        except Exception as e: logging.error(f"[DateBudgetNode] Error: {e}", exc_info=True); error_msg = f"Error: {e}"; return {"date_budget_summary": error_msg, "error_message": error_msg}


    def process_destination_node(self, state: TravelPlanState) -> Dict[str, Any]:
        logging.info("[DestinationNode] Running...")
        parsed_info = state.get('parsed_request'); calculated_dates = state.get('calculated_dates')
        origin_city = state.get('origin')
        logging.debug(f"[DestinationNode] Incoming State['parsed_request']: {json.dumps(parsed_info, indent=2, ensure_ascii=False)}")
        logging.debug(f"[DestinationNode] Incoming State['calculated_dates']: {json.dumps(calculated_dates, indent=2, ensure_ascii=False)}")
        logging.debug(f"[DestinationNode] Incoming State['origin']: {origin_city}")
        if not parsed_info or not calculated_dates:
            logging.warning("[DestinationNode] Critical information missing (parsed_info or calculated_dates).")
            missing = []
            if not parsed_info: missing.append("parsed_info")
            if not calculated_dates: missing.append("calculated_dates")
            error_msg = f"Skipped: Missing critical info: {', '.join(missing)}."
            return {"destination_summary": error_msg, "error_message": state.get("error_message") or error_msg}

        destination_city = parsed_info.get('destination')
        start_date = calculated_dates.get('start_date')
        end_date = calculated_dates.get('end_date')
        budget_amount_ref = parsed_info.get('budget_amount', 'N/A')
        budget_currency_ref = parsed_info.get('budget_currency', '')

        logging.debug(f"[DestinationNode] Extracted destination_city: {destination_city}")
        logging.debug(f"[DestinationNode] Extracted start_date: {start_date}")
        logging.debug(f"[DestinationNode] Extracted end_date: {end_date}")
        logging.debug(f"[DestinationNode] Extracted budget_amount_ref: {budget_amount_ref}")
        logging.debug(f"[DestinationNode] Extracted budget_currency_ref: {budget_currency_ref}")

        required_sub_keys = {"destination": destination_city, "start_date": start_date, "end_date": end_date}
        missing_sub_keys = [key for key, value in required_sub_keys.items() if not value]
        if missing_sub_keys:
            logging.warning(f"[DestinationNode] Required sub-information missing: {', '.join(missing_sub_keys)}.")
            error_msg = f"Skipped: Missing sub-keys: {', '.join(missing_sub_keys)}."
            current_error = state.get("error_message")
            new_error = f"{current_error + '; ' if current_error else ''}{error_msg}"
            return {"destination_summary": error_msg, "error_message": new_error}

        destination_query = f"""
        Please collect detailed travel information for the following trip and provide the result as an English summary:
        - Origin: {origin_city or 'Not specified'}
        - Destination: {destination_city}
        - Start Date: {start_date}
        - End Date: {end_date}
        - Budget Information (for reference): {budget_amount_ref} {budget_currency_ref}

        Tasks & Output Structure (Use EXACT Headings):
        1. Use `search_city_info` for {destination_city}.
        2. Use `get_weather_forecast` for {destination_city} between {start_date} - {end_date}.
        3. Use `Google Hotels_with_tavily` for {destination_city} for dates {start_date} to {end_date}. Note limitations.
        4. Use `get_tomtom_map_url` with `city_name`='{destination_city}'.
        5. Combine results under: 'City Information', 'Weather/Clothing Recommendations', 'Hotel Options', 'Map View'. Include map URL if available. Respond ONLY in English. If a tool fails, note it politely and continue.
        """

        logging.info(f"--- [DestinationNode] Beginning of SYNCHRONOUS Prompt to be sent to Destination Agent ---")
        logging.info(destination_query)
        logging.info(f"--- [DestinationNode] End of SYNCHRONOUS Prompt to be sent to Destination Agent ---")

        logging.info("[DestinationNode] Calling Destination Agent (SYNCHRONOUS)...")
        try:
            agent_input = {"input": destination_query}
            response = self.destination_agent.invoke(agent_input)

            logging.debug(f"[DestinationNode] Agent Raw Response: {response}")

            summary = extract_agent_output_text(response.get("output", "Destination summary error."))
            logging.info(f"[DestinationNode] Agent Summary Result: {summary}")

            error_in_summary = False
            if isinstance(summary, str):
                summary_lower = summary.lower()
                if "error" in summary_lower or "hata" in summary_lower or "lütfen" in summary_lower or "belirtiniz" in summary_lower or "eksik" in summary_lower or "summary error" in summary_lower:
                    error_in_summary = True
                    logging.warning(f"[DestinationNode] Potential error/missing information detected in agent response: {summary}")

            current_error = state.get("error_message")
            new_error = current_error
            if error_in_summary:
                error_msg = f"Destination Agent Error/Incomplete: {summary}"
                new_error = f"{current_error + '; ' if current_error else ''}{error_msg}"

            return {"destination_summary": summary, "error_message": new_error}

        except Exception as e:
            logging.error(f"[DestinationNode] Error calling Agent: {e}", exc_info=True)
            error_msg = f"Error calling Destination Agent: {type(e).__name__}"
            current_error = state.get("error_message")
            new_error = f"{current_error + '; ' if current_error else ''}{error_msg}"
            return {"destination_summary": f"Error: {type(e).__name__}", "error_message": new_error}


    def compile_final_plan_node(self, state: TravelPlanState) -> Dict[str, Any]: 
        logging.info("[CompileNode] Running...")

        logging.debug(f"--- [CompileNode] Beginning of Incoming State ---")
        logging.debug(f"User Query: {state.get('user_query')}")
        logging.debug(f"Parsed Request: {state.get('parsed_request')}")
        logging.debug(f"Calculated Dates: {state.get('calculated_dates')}")
        logging.debug(f"Date Budget Summary: {state.get('date_budget_summary')}")
        logging.debug(f"Destination Summary: {state.get('destination_summary')}")
        logging.debug(f"Error Message: {state.get('error_message')}")
        logging.debug(f"--- [CompileNode] End of Incoming State ---")

        error_msg = state.get("error_message")
        if error_msg and ("parse" in error_msg.lower() or "date calculation failed" in error_msg.lower() or "ayrıştırılamadı" in error_msg.lower()):
            logging.error(f"[CompileNode] Cannot compile plan due to critical error: {error_msg}"); return {"final_plan": f"Plan could not be created. Basic information could not be parsed or dates could not be calculated. Error: {error_msg}"}

        date_budget_summary = state.get('date_budget_summary', 'Budget/Date Summary Not Available') 
        destination_summary = state.get('destination_summary', 'Destination Information Summary Not Available') 
        parsed_info = state.get('parsed_request', {})
        calculated_dates = state.get('calculated_dates', {})
        start_date = calculated_dates.get('start_date', '?')
        end_date = calculated_dates.get('end_date', '?')

        logging.debug(f"[CompileNode] Received Date Budget Summary: {date_budget_summary[:200]}...") 
        logging.debug(f"[CompileNode] Received Destination Summary: {destination_summary[:200]}...") 

        if date_budget_summary == 'Budget/Date Summary Not Available' or destination_summary == 'Destination Information Summary Not Available':
            logging.error("[CompileNode] Critical summary information not found in state!")

        destination_summary_for_prompt = destination_summary
        if error_msg and "Destination Agent Error" in error_msg: 
            destination_summary_for_prompt = f"(Note: Problem occurred while getting destination information: {destination_summary})"
        elif error_msg and "DateBudget Agent Error" in error_msg: 
            pass

        final_prompt = f"""
        Create a final travel plan summary in ENGLISH for the user using the information below.

        User Request: {state.get('user_query', 'N/A')}
        Parsed Info: {json.dumps(parsed_info, ensure_ascii=False, indent=2)}
        Calculated Dates: {start_date} - {end_date} (Duration: {parsed_info.get('duration_days', '?')} days)
        Origin: {parsed_info.get('origin', 'Not specified')}

        --- Date/Budget Summary ---
        {date_budget_summary}
        --- End Date/Budget Summary ---

        --- Destination Summary (Includes City Info, Weather, Hotels, Map URL) ---
        {destination_summary_for_prompt}
        --- End Destination Summary ---

        Task: Synthesize all this information to create a plan with the following ENGLISH headings:
        1. Trip Summary (Origin, Destination, Dates, Duration - From Parsed Information)
        2. Budget & Exchange Rate Info (Should be taken from Date/Budget Summary)
        3. Weather & Clothing Recommendations (Should be taken from Destination Summary)
        4. City & Trip Information (Should be taken from Destination Summary)
        5. Accommodation Recommendations (Should be taken from Destination Summary, note limitations)
        6. Map View (Extract and include Map URL from Destination Summary)

        If information is missing or an error occurred in previous steps (as indicated in the summaries), politely note this. Only compile, don't call new tools. Response should be ONLY IN ENGLISH.
        """
        logging.info(f"--- [CompileNode] Beginning of SYNCHRONOUS Prompt to be sent to Coordinator Agent ---")
        logging.info(final_prompt)
        logging.info(f"--- [CompileNode] End of SYNCHRONOUS Prompt to be sent to Coordinator Agent ---")

        logging.info("[CompileNode] Calling Coordinator Agent (SYNCHRONOUS)...")
        try:
            agent_input = {"input": final_prompt}
            final_response = self.coordinator_agent.invoke(agent_input)

            logging.debug(f"[CompileNode] Coordinator Raw Response: {final_response}")

            final_output = extract_agent_output_text(final_response.get("output", "Final plan generation failed."))
            logging.info(f"[CompileNode] Agent Final Plan: {final_output}")
            return {"final_plan": final_output}
        except Exception as e:
            logging.error(f"[CompileNode] ERROR calling Coordinator: {e}", exc_info=True)
            return {"final_plan": f"Error occurred with Coordinator Agent while compiling plan: {type(e).__name__}."}

    def decide_after_parsing(self, state: TravelPlanState) -> str:
        if state.get("error_message") and ("parse" in state["error_message"].lower() or "ayrıştırılamadı" in state["error_message"].lower()):
             return "compile_final_plan" 
        else: return "calculate_dates" 
    def decide_after_dates(self, state: TravelPlanState) -> str:
        if state.get("error_message") and "date calculation failed" in state["error_message"].lower():
             return "compile_final_plan" 
        else: return "process_date_budget"
        
    def build_graph(self) -> Runnable:
        logging.info("Creating LangGraph workflow (SYNCHRONOUS nodes)...")
        workflow = StateGraph(TravelPlanState)

        workflow.add_node("parse_request", self.parse_request_node)
        workflow.add_node("calculate_dates", self.calculate_dates_node)
        workflow.add_node("process_date_budget", self.process_date_budget_node)
        workflow.add_node("process_destination", self.process_destination_node)
        workflow.add_node("compile_final_plan", self.compile_final_plan_node)

        workflow.set_entry_point("parse_request")

        workflow.add_conditional_edges(
            "parse_request",
            path=self.decide_after_parsing,
            path_map={
                "calculate_dates": "calculate_dates",
                "compile_final_plan": "compile_final_plan",
            }
        )
        workflow.add_conditional_edges(
            "calculate_dates",
            path=self.decide_after_dates,
            path_map={
                "process_date_budget": "process_date_budget",
                "compile_final_plan": "compile_final_plan",
            }
        )

        workflow.add_edge("process_date_budget", "process_destination")
        workflow.add_edge("process_destination", "compile_final_plan")
        workflow.add_edge("compile_final_plan", END)

        try:
             app = workflow.compile(checkpointer=MemorySaver())
             logging.info("LangGraph workflow successfully compiled (Travel System - Synchronous, with Checkpointer).")
        except ImportError:
             logging.error("Could not import MemorySaver! Cannot continue without checkpointing.")
             raise
        except Exception as e:
             logging.error(f"Error compiling graph (with checkpointer): {e}", exc_info=True)
             logging.warning("Compiling without checkpointer...")
             app = workflow.compile() 

        return app

    def process_query(self, user_query: str) -> str:
        logging.info(f"process_query (Synchronous) called: {user_query}")
        initial_state = {"user_query": user_query}
        import uuid
        config = {"configurable": {"thread_id": f"travel-sync-thread-{uuid.uuid4()}"}}
        try:
            final_state = self.app.invoke(initial_state, config=config)
            final_plan_output = final_state.get("final_plan", "Error: Could not get final plan from state.")
            if isinstance(final_plan_output, str) and ("error occurred" in final_plan_output.lower() or "oluştu" in final_plan_output.lower() or "failed" in final_plan_output.lower()):
                logging.error(f"TravelPlanningSystem (Synchronous) returned error: {final_plan_output}")
            else:
                logging.info("TravelPlanningSystem (Synchronous) completed successfully.")
            return final_plan_output
        except Exception as e:
            logging.error(f"TravelPlanningSystem (Synchronous) process_query error: {e}", exc_info=True)
            return f"System error occurred: {type(e).__name__}"