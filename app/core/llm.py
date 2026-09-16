# app/core/llm.py

import os
import logging
from langchain_anthropic import ChatAnthropic
from typing import Optional, Dict, Tuple, Any
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

try:
    from configs import api_config
except ImportError:
    logging.error("Module configs.api_config not found! Environment variables might not be loaded.")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')

llm_instances: Dict[Tuple, ChatAnthropic] = {}

def get_llm(
    model_name: str = "claude-sonnet-5",
    temperature: float = 0.5,
    max_output_tokens: Optional[int] = 2048,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    **kwargs: Any
) -> Optional[ChatAnthropic]:
    # Note: temperature/top_p/top_k are accepted for call-site compatibility but are
    # NOT forwarded to Claude below. Claude models run extended thinking by default,
    # and the API rejects sampling params while thinking is active. Rather than
    # disabling thinking (which has documented failure modes on some models), we
    # leave thinking on and simply drop these params.

    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_api_key:
        logging.error("Environment variable 'ANTHROPIC_API_KEY' not found or is empty!")
        return None

    cache_key = (
        model_name, temperature, max_output_tokens, top_p, top_k,
        tuple(sorted(kwargs.items()))
    )

    if cache_key in llm_instances:
        logging.debug(f"Returning LLM instance from cache (Config: {cache_key})")
        return llm_instances[cache_key]

    logging.info(f"Creating new LLM instance: Model={model_name}...")

    try:
        llm = ChatAnthropic(
            model=model_name,
            api_key=anthropic_api_key,
            max_tokens=max_output_tokens,
            **kwargs
        )
        llm_instances[cache_key] = llm
        logging.info("LLM instance successfully created and cached.")
        return llm
    except Exception as e:
        logging.error(f"Error occurred while creating LLM instance: {e}", exc_info=True)
        return None
