# app/tools/external_apis.py

import wikipedia
import logging
from pathlib import Path
from typing import Optional
from tavily import TavilyClient
from langchain.tools import Tool
import sys
from pathlib import Path
import os 
from langchain_community.tools import DuckDuckGoSearchRun

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY") 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')

tavily_client: Optional[TavilyClient] = None 
tavily_error_message: Optional[str] = None

if TAVILY_API_KEY:
    try:
        tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
        logging.info("Tavily API client successfully started.")
    except Exception as e:
        tavily_error_message = f"Error while starting Tavily API client: {e}."
        logging.error(tavily_error_message, exc_info=True)
        tavily_client = None
else:
    logging.warning("Environment variable 'TAVILY_API_KEY' not found or empty. Tavily client could not be started.")
    tavily_error_message = "Tavily API key not set."

def search_wikipedia(query: str, lang: str = "en", sentences: int = 5) -> str:
    logging.info(f"Searching for '{query}' in Wikipedia (language={lang})...")
    wikipedia.set_lang(lang)
    try:
        page = wikipedia.page(query, auto_suggest=False)
        summary = wikipedia.summary(query, sentences=sentences, auto_suggest=False)
        logging.info(f"Wikipedia summary found (Page: {page.title}, URL: {page.url}).")
        return f"Wikipedia Result ({page.title}):\n{summary}\n\nSource: {page.url}"
    except wikipedia.exceptions.PageError:
        logging.warning(f"'{query}' page not found in Wikipedia.")
        return f"Sorry, I couldn't find a page for '{query}' on Wikipedia."
    except wikipedia.exceptions.DisambiguationError as e:
        options_preview = ", ".join(e.options[:5])
        logging.warning(f"Wikipedia query '{query}' has multiple meanings: {options_preview}...")
        return f"'{query}' query has multiple meanings (e.g., {options_preview}...). Please clarify your query."
    except Exception as e:
        logging.error(f"Unexpected error during Wikipedia search: {e}", exc_info=True)
        return f"An error occurred during the Wikipedia search: {e}"

wikipedia_tool = Tool(
    name="WikipediaSearch",
    func=search_wikipedia,
    description="Used to get encyclopedic information about a specific topic, person, place, or event. It is good for definitions and general information. It performs searches in English."
)

def search_web_tavily(query: str, max_results: int = 5) -> str:
    if not tavily_client:
        return tavily_error_message or "Tavily API client is unavailable."
    logging.info(f"Searching for '{query}' on the web with Tavily (max_results={max_results})...")
    try: 
        response = tavily_client.search(query=query, search_depth="basic", max_results=max_results)
        results = response.get('results', [])
        if results:
            formatted_results = []
            for res in results:
                formatted_results.append(
                    f"Title: {res.get('title', 'N/A')}\n"
                    f"URL: {res.get('url', 'N/A')}\n"
                    f"Summary: {res.get('content', 'N/A')}"
                )
            logging.info(f"Tavily found {len(results)} results.")
            return "\n\n---\n\n".join(formatted_results)
        else:
            logging.warning("Tavily search returned no results.")
            return "Web search (Tavily) returned no results for this query."
    except Exception as e:
        logging.error(f"Error during Tavily API call: {e}", exc_info=True)
        return f"An error occurred during the web search (Tavily): {e}"

def search_web_duckduckgo(query: str) -> str:
    logging.info(f"Searching for '{query}' on the web with DuckDuckGo...")
    try:
        ddg_search = DuckDuckGoSearchRun()
        results = ddg_search.run(query)
        if results and "No good DuckDuckGo Search Result" not in results:
            logging.info("DuckDuckGo search returned results.")
            return results
        else:
            logging.warning("DuckDuckGo search returned no meaningful results.")
            return "Web search (DuckDuckGo) returned no results for this query."
    except Exception as e:
        logging.error(f"Error during DuckDuckGo web search: {e}", exc_info=True)
        return f"An error occurred during the web search (DuckDuckGo): {e}"

def search_web(query: str) -> str:
    logging.info(f"Starting general web search: '{query}'")
    if tavily_client:
        logging.debug("Attempting web search with Tavily...")
        return search_web_tavily(query)
    else:
        logging.debug("Tavily is unavailable, trying web search with DuckDuckGo...")
        return search_web_duckduckgo(query)

web_search_tool = Tool(
    name="WebSearch",
    func=search_web,
    description="Performs web searches for up-to-date events, news, weather, stock prices, or specific information not available on Wikipedia. Useful for getting the latest information."
)
