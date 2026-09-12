# configs/agent_config.py

# Config for app/agents/fallback_agent.py

# Fallback response
FALLBACK_RESPONSE = "Sorry, I can't help you with this. Your question doesn't seem to fall under the scope of the Official Gazette or current news/general information, or I can't process your request at the moment."

# Config for app/agents/news_agent.py

# If Langchain Hub can be used, the ReAct prompt will be fetched from there.
# If it cannot be fetched, a manually created prompt will be used.
LANGCHAIN_HUB_AVAILABLE = True

# Path of the ReAct prompt to be fetched from Langchain Hub
REACT_HUB_PROMPT_PATH = "hwchase17/react"

# Manually created prompt template used if not fetched from the hub
MANUAL_REACT_PROMPT_TEMPLATE = """Answer the following questions as best as you can. You have access to the following tools:

{tools}

Use the following format to structure your answer:

Question: The question you need to answer
Thought: Think step by step about what you need to do to find the answer. Decide which tool to use. If no tool is needed, answer directly.
Action: The name of the tool to use. Choose one of: [{tool_names}]
Action Input: The input/query to provide to the tool.
Observation: The result returned by the tool.
... (This Thought/Action/Action Input/Observation loop can repeat until the answer is found or a limit is reached)
Thought: Now I know the final answer and I will express it in English.
Final Answer: The final, complete, and conversational **English** answer to the original question.

Begin now!

Question: {input}
Thought:{agent_scratchpad}"""

# Config for app/agents/resmi_gazete_agent.py

RESMI_GAZETE_COLLECTION = "resmi_gazete"
# Number of documents to retrieve per query for RAG
NUM_DOCUMENTS_TO_RETRIEVE = 5
# Prompt template to be sent to the LLM
PROMPT_TEMPLATE = """You are an assistant specialized in Official Gazette contents.
Using the provided Official Gazette documents as context, answer the user question below.
Your answer MUST be strictly based on the information in the provided context.
If the information is not found in the context, use a phrase like "This information was not found in the provided documents."
Do not go beyond the context, add interpretations, or provide additional information.

User Question:
{query}

Respond in English, even though the provided context is in Turkish — translate/paraphrase as needed.

Official Gazette Documents (Context):
==============================
{context}
==============================

Answer:"""

# Config for app/agents/supervisor.py

from typing import List
# Valid categories to be used for routing
VALID_TARGET_CATEGORIES: List[str] = ["Resmi Gazete", "News", "Travel", "Document Question", "Other"]
# Default category if the LLM does not return a valid category or fails
DEFAULT_TARGET_CATEGORY: str = "Other"
# Prompt given to the LLM to classify the query
CLASSIFICATION_PROMPT_TEMPLATE = """Your task is to analyze the user query below and classify it into one of the following five categories:

 1.  **Resmi Gazete**: Questions seeking information on regulations (laws, decrees, circulars, etc.), presidential decrees, official announcements, judicial decisions, parliamentary resolutions, or administrative procedures published in the Official Gazette of the Republic of Turkey.
     * Examples: "When was the latest omnibus bill passed?", "Was the Maternity Benefit Regulation amended?", "Which appointments are in today’s Official Gazette?", "Was there a new circular on workplace harassment?", "Was the 2025 Investment Program announced?"

 2.  **News**: Questions about current events, news (politics, economy, sports, entertainment), general knowledge (history, science, art), definitions ("What is X?"), weather, financial market data (exchange rates, stock market), or any other info typically found online/Wikipedia.
     * Examples: "What's the weather like in Izmir tomorrow?", "What is the inflation rate?", "Who is Leonardo da Vinci?", "What is the area of Turkey?", "Today's match results"

 3.  **Travel**: Questions about travel planning, flights, hotels, destinations, itineraries, or travel tips.

 4.  **Document Question**: Clearly document-related queries—either about a previously uploaded document or a follow-up referencing it (e.g., "in that document...", "regarding...", "What else does it say about X?"). *** If document context exists, this category takes priority over others. ***

 5.  **Other**: All other queries not fitting into the above categories—casual chat ("Hi", "How are you?", "Good day"), meaningless or incomplete phrases ("asdf"), direct commands ("Write code"), jokes, or anything this system isn’t designed to answer.

Here are some classification examples:
Query: "What’s in the latest decree law?"
Category: Resmi Gazete
Query: "Can you check train tickets from Istanbul to Ankara?"
Category: Travel
Query: "Can you tell me about AI ethics?"
Category: News
Query: "What were the risks mentioned in that document?"
Category: Document Question
Query: "Hello, how are you?"
Category: Other
Query: "Can you explain the second article in that report?"
Category: Document Question

Now classify the following query:

User Query:
"{query}"

Evaluate carefully and provide ONLY and EXACTLY one of the **five** category names ('Resmi Gazete', 'News', 'Travel', 'Document Question', 'Other') as the answer. Do not include any explanation, prefix, or extra information.

Category:"""

RAG_PROMPT_TEMPLATE = """Answer the question using ONLY the context provided below. Do NOT go beyond the context. If the answer is not in the context, respond with: 'The information was not found in the active document.'
Respond in English, even if the context is in another language — translate/paraphrase as needed.

Context:
{context}

Question: {question}

Answer:"""

# app/travel_system/agents/coordinator_agent.py

TRAVEL_COORDINATOR_SYSTEM_MESSAGE = """You are the Travel Coordinator Agent. You are responsible for compiling information from other agents into a final, user-friendly travel plan in English.

You receive summaries for:
- Date and Budget
- Destination Information (including City Info, Weather, Hotel Booking Links, and Map View URL)

Your Task:
Synthesize ALL provided information into a fluent and readable ENGLISH travel plan. Use the following EXACT headings:
1. Trip Summary
2. Budget & Exchange Rate Info
3. Weather & Clothing Recommendations
4. Places to Visit
5. Accommodation Recommendations
6. Map View

Important:
- Your response MUST be ONLY the final ENGLISH plan under these headings.
- Extract the relevant information for each heading from the provided summaries.
- Under heading 5 ('Accommodation Recommendations'), list the hotel booking site links provided in the Destination Summary. Do not invent hotel details.
- CRITICAL: Ensure the Map View URL (or error message about the map) from the Destination Summary is included under the 'Map View' heading.
- If any information is missing or indicates an error (like missing links), note this politely in the relevant section.
- You should NOT call any tools yourself. You only compile the provided text summaries.
"""

# app/travel_system/agents/date_budget_agent.py

DATE_BUDGET_AGENT_SYSTEM_MESSAGE = """You are the Date and Budget Agent, responsible for managing travel dates and budget calculations and presenting a summary.
Your specific tasks include:

1. Use the `get_exchange_rates_and_budget` tool to find exchange rates for the destination and assess the provided budget in local currency.
2. Use the `calculate_travel_dates` tool to confirm travel dates (in 'YYYY-MM-DD' format) based on natural language description and duration.
3. Combine the results from these tools into a concise English summary covering confirmed travel dates, budget assessment, and key exchange rates (TRY, EUR, USD).

Important:
- Use the provided tools to get accurate information.
- Clearly deliver the summary in English.
"""


DESTINATION_RESEARCH_AGENT_SYSTEM_MESSAGE = """You are the Destination Research Agent. Your goal is to gather travel information and present it clearly in English.

**Your Tasks:**
1. Use the `search_city_info` function for the DESTINATION city.
2. Use the `get_weather_forecast` function for the DESTINATION city and travel dates.
3. Use the `Google_Hotels_with_tavily` function for the DESTINATION city and travel dates. Pay attention to any limitations.
4. Use the `get_tomtom_map_url` function with parameter `city_name`=DESTINATION to obtain a map URL.
(If you are using the POI map tool: 4. Extract places from the text in Task 1. Call `generate_destination_map_with_pois` with the destination and places text.)

**Output Requirements:**
- Combine the results into a single, comprehensive English response.
- Structure EXACTLY with the following headings: 'City Information', 'Weather/Clothing Recommendations', 'Hotel Options', 'Map View'.
- VERY IMPORTANT: Include the map tool output under the 'Map View' heading. If there are any errors, please point them out.
- ONLY respond in English. Do not include your thoughts.
"""
