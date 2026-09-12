# app/core/llm.py

import os
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
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

llm_instances: Dict[Tuple, ChatGoogleGenerativeAI] = {}

def get_llm(
    model_name: str = "gemini-flash-latest",
    temperature: float = 0.5, 
    max_output_tokens: Optional[int] = 2048,
    top_p: Optional[float] = None, 
    top_k: Optional[int] = None, 
    **kwargs: Any
) -> Optional[ChatGoogleGenerativeAI]:

    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        logging.error("Environment variable 'GEMINI_API_KEY' not found or is empty!")
        return None 

    cache_key = (
        model_name, temperature, max_output_tokens, top_p, top_k,
        tuple(sorted(kwargs.items()))
    )

    if cache_key in llm_instances:
        logging.debug(f"Returning LLM instance from cache (Config: {cache_key})")
        return llm_instances[cache_key]

    logging.info(f"Creating new LLM instance: Model={model_name}, Temp={temperature}...")

    try:
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=gemini_api_key,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            top_p=top_p,
            top_k=top_k,
            **kwargs
        )
        llm_instances[cache_key] = llm
        logging.info("LLM instance successfully created and cached.")
        return llm
    except Exception as e:
        logging.error(f"Error occurred while creating LLM instance: {e}", exc_info=True)
        return None
