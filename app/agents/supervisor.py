# app/agents/supervisor.py

import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from app.core.llm import get_llm
from configs.agent_config import VALID_TARGET_CATEGORIES, DEFAULT_TARGET_CATEGORY, CLASSIFICATION_PROMPT_TEMPLATE

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')

# Function used for query classification
def classify_query(state: Dict[str, Any]) -> Dict[str, Any]:

    logging.info("Supervisor Agent is running (Query Classification)...")
    query: Optional[str] = state.get("query")
    # Initially, we set it to the default category
    classification_decision = DEFAULT_TARGET_CATEGORY
    source_info = "Supervisor"
    # Pre-fallback LLM output and token usage, exposed for eval purposes
    # (unused by the graph/UI, which only reads "classification")
    raw_llm_output = None
    usage = None

    # If there is no valid query, set category to 'Other'
    if not query or not query.strip():
        logging.warning("Supervisor: No valid query found in state or the query is empty. Category set to 'Other'.")
        classification_decision = "Other"
        source_info += " (Error: Empty Query)"
    # If the query is valid, proceed with classification
    else:
        logging.info(f"Query to be classified: '{query}'")
        try:
            # We call the LLM with temperature 0.0 for deterministic output
            llm = get_llm(temperature=0.0)

            # Create the prompt
            category_list_str = ", ".join([f"'{cat}'" for cat in VALID_TARGET_CATEGORIES])
            prompt = CLASSIFICATION_PROMPT_TEMPLATE.format(
                query=query,
                category_list_str=category_list_str
            )
            logging.debug("Classification prompt is being sent to the LLM...")
            # Send the prompt to the LLM
            response = llm.invoke(prompt)
            llm_output = response.content.strip()
            raw_llm_output = llm_output
            usage = getattr(response, "usage_metadata", None)
            logging.info(f"Raw LLM classification output: '{llm_output}'")

            # Ensure LLM only returns a valid category name
            if llm_output in VALID_TARGET_CATEGORIES:
                classification_decision = llm_output
                logging.info(f"Query successfully classified: '{classification_decision}'")
                source_info += " (LLM Successful)"
            else:
                # If the LLM output doesn't match exactly, check if a valid category is inside the response
                found_category = None
                for valid_cat in VALID_TARGET_CATEGORIES:
                    if valid_cat in llm_output:
                        found_category = valid_cat
                        logging.warning(f"LLM output '{llm_output}' does not exactly match, but valid category '{found_category}' found within. Using this category.")
                        break
                if found_category:
                    classification_decision = found_category
                    source_info += " (LLM Partially Successful)"
                else:
                    # If no valid category is found at all, revert to default
                    logging.error(f"LLM output '{llm_output}' does not match or contain any valid categories ({VALID_TARGET_CATEGORIES})! Default category '{DEFAULT_TARGET_CATEGORY}' will be used.")
                    classification_decision = DEFAULT_TARGET_CATEGORY
                    source_info += " (Error: Invalid LLM Output)"
        # If an error occurs during LLM call, fallback to default
        except Exception as e:
            logging.error(f"Error during query classification: {e}", exc_info=True)
            classification_decision = DEFAULT_TARGET_CATEGORY
            source_info += f" (Error: {type(e).__name__})"

    # Return with the 'classification' key added to the LangGraph state.
    # raw_llm_output/usage are additive, eval-only keys: the graph and UI
    # only ever read "classification", so these are safe to ignore elsewhere.
    logging.info(f"Supervisor completed. Final Classification Result: '{classification_decision}'")
    return {
        "classification": classification_decision,
        "raw_llm_output": raw_llm_output,
        "usage": usage,
    }
