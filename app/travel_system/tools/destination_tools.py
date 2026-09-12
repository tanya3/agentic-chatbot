# app/travel_system/tools/destination_tools.py

import os
import requests
import json
import logging
from datetime import datetime
from urllib.parse import quote
from typing import Optional, Dict, Any
from langchain_core.tools import tool
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults
from app.core.llm import get_llm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')

@tool
def search_city_info(city_name: str) -> str:
    """
    General information about the given city, tourist attractions, etc.
    Searches using the Google Serper API.
    """
    serper_api_key = os.getenv("SERPER_API_KEY")
    if not serper_api_key:
        logging.error("SERPER_API_KEY environment variable not found.")
        return "Error: City information search API key (Serper) not found."

    logging.info(f"Searching for information about '{city_name}' with Google Serper API...")
    try:
        search = GoogleSerperAPIWrapper(serper_api_key=serper_api_key)
        query = f"{city_name} general information, tourist attractions, popular places"
        results = search.run(query)
        logging.info(f"Results received from Google Serper API.")
        # Note if there are no results
        return results if results else f"No information found about '{city_name}' with Google Serper."
    except Exception as e:
        logging.error(f"Error during Google Serper API search: {e}", exc_info=True)
        return f"An error occurred while searching for city information: {e}"

def get_coordinates(city_name: str, api_key: str) -> Dict[str, Any]:
    """Gets latitude and longitude for a city using OpenWeatherMap Geocoding API."""

    logging.debug(f"Getting coordinates for '{city_name}' with OpenWeatherMap Geocoding...")
    base_url = "http://api.openweathermap.org/geo/1.0/direct"
    params = {
        'q': city_name,
        'limit': 1,
        'appid': api_key
    }
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        if data and isinstance(data, list) and len(data) > 0:
             lat = float(data[0].get('lat', 0.0))
             lon = float(data[0].get('lon', 0.0))
             logging.debug(f"Coordinates found: Lat={lat}, Lon={lon}")
             return {"lat": lat, "lon": lon}
        else:
             logging.warning(f"OpenWeatherMap Geocoding API couldn't find the city '{city_name}' or returned an empty response.")
             return {"error": f"OpenWeatherMap Geocoding API couldn't find the city '{city_name}'."}
    except requests.exceptions.RequestException as e:
        logging.error(f"Error connecting to OpenWeatherMap Geocoding API: {e}", exc_info=True)
        return {"error": f"Weather coordinates could not be retrieved (connection error): {e}"}
    except (ValueError, KeyError, IndexError) as e:
        logging.error(f"Error processing OpenWeatherMap Geocoding API response: {e}", exc_info=True)
        return {"error": f"Weather coordinates could not be retrieved (response format error): {e}"}
    except Exception as e:
        logging.error(f"Unexpected error while getting coordinates: {e}", exc_info=True)
        return {"error": f"An unexpected error occurred while getting coordinates: {e}"}

@tool
def get_weather_forecast(city_name: str, start_date_str: str, end_date_str: str) -> str:
    """
    OpenWeatherMap API for specified city and dates (YYYY-MM-DD) (5 days)
    receives the weather forecast using LLM and provides an outfit recommendation in English using LLM.
    """
    api_key = os.getenv("OPENWEATHERMAP_API_KEY")
    if not api_key:
        logging.error("OPENWEATHERMAP_API_KEY environment variable not found.")
        return "Error: Weather API key not found."

    logging.info(f"Searching for weather forecast for '{city_name}' between {start_date_str} - {end_date_str} with OpenWeatherMap...")
    coords = get_coordinates(city_name, api_key)
    if "error" in coords:
        logging.error(f"Coordinates could not be retrieved: {coords['error']}")
        return f"Error: {coords['error']}"

    base_url = "http://api.openweathermap.org/data/2.5/forecast"
    params = {
        'lat': coords['lat'],
        'lon': coords['lon'],
        'appid': api_key,
        'units': 'metric',
        'lang': 'en'
    }
    forecast_summary = f"Weather forecast summary could not be retrieved ({city_name})."
    relevant_forecasts_str = "Detailed forecast not found."

    try:
        response = requests.get(base_url, params=params)
        logging.debug(f"OpenWeatherMap API Response Code: {response.status_code}")
        response.raise_for_status()
        weather_data = response.json()
        logging.debug(f"OpenWeatherMap Raw Response: {json.dumps(weather_data, indent=2, ensure_ascii=False)}")

        if str(weather_data.get("cod")) != "200":
             message = weather_data.get("message", "Unknown API error")
             logging.error(f"OpenWeatherMap API Error (cod != 200): {message}")
             return f"Weather forecast could not be retrieved: {message}"

        forecast_summary = f"Weather Forecast Summary for {city_name} from {start_date_str} to {end_date_str}:\n"
        try:
            start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
            end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
        except ValueError:
            logging.error(f"Invalid date format: {start_date_str} or {end_date_str}")
            return "Error: Invalid date format (YYYY-MM-DD expected)."

        relevant_forecasts = []
        if 'list' in weather_data and isinstance(weather_data['list'], list):
            for forecast in weather_data['list']:
                 try:
                     forecast_dt = datetime.fromtimestamp(int(forecast.get('dt', 0)))
                     forecast_date = forecast_dt.date()
                     if start_date <= forecast_date <= end_date and 11 <= forecast_dt.hour <= 14:
                         desc = forecast.get('weather',[{}])[0].get('description','no information').capitalize()
                         temp = forecast.get('main',{}).get('temp','?')
                         feels = forecast.get('main',{}).get('feels_like','?')
                         hum = forecast.get('main',{}).get('humidity','?')
                         wind = forecast.get('wind',{}).get('speed','?')
                         relevant_forecasts.append(
                             f"- {forecast_date.strftime('%Y-%m-%d %A')}: {desc}, "
                             f"Temperature: {temp}°C (Feels like: {feels}°C), "
                             f"Humidity: %{hum}, Wind: {wind} m/s"
                         )
                 except (KeyError, IndexError, ValueError, TypeError) as item_err:
                     logging.warning(f"Error processing weather list item skipped: {item_err} - Data: {forecast}")
                     continue

            relevant_forecasts = sorted(list(set(relevant_forecasts)))

        if not relevant_forecasts:
            relevant_forecasts_str = "Detailed forecast for the specified dates not found (API provides 5-day data).\n"
            logging.warning(relevant_forecasts_str)
        else:
            relevant_forecasts_str = "\n".join(relevant_forecasts) + "\n"
            logging.info("Relevant weather forecasts processed.")

        forecast_summary += relevant_forecasts_str
        logging.info("Requesting clothing suggestion from LLM for weather summary...")
        llm_instance = get_llm(temperature=0.3)
        if not llm_instance:
            logging.error("LLM instance could not be obtained for clothing suggestion.")
            suggestion_text = "Clothing suggestions could not be generated due to system error."
        else:
            prompt = f"""Given the following weather summary, can you provide practical and brief clothing suggestions in English for a traveler? Focus only on clothing suggestions, don't repeat the weather forecast. Example: "Pack layered clothing, a light jacket, and an umbrella." etc.

Weather Summary:
{relevant_forecasts_str}

Clothing Suggestions:"""
            logging.debug(f"LLM Prompt for clothing suggestion:\n{prompt}")
            try:
                response = llm_instance.invoke(prompt)
                suggestion_text = response.content.strip()
                logging.info(f"Clothing suggestion received from LLM: {suggestion_text}")
            except Exception as llm_err:
                logging.error(f"Error getting clothing suggestion from LLM: {llm_err}", exc_info=True)
                suggestion_text = f"An LLM error occurred while getting clothing suggestions."

        final_output = forecast_summary.strip() + "\n\nClothing Suggestions:\n" + suggestion_text
        logging.info("get_weather_forecast tool completed and returning string result.")
        return final_output

    except requests.exceptions.RequestException as e:
        logging.error(f"Error connecting to OpenWeatherMap API: {e}", exc_info=True)
        return f"Error: Could not connect to weather API: {e}"
    except Exception as e:
        logging.error(f"Unexpected error while retrieving or processing weather: {e}", exc_info=True)
        return f"Error: An unexpected problem occurred while retrieving or processing weather: {e}"

@tool
def search_hotel_booking_links(destination: str, start_date: str, end_date: str, budget_info: Optional[str] = None) -> str:
    """
    Uses Tavily Search API to find links to popular hotel booking websites
    (like Google Hotels, Booking.com, Expedia) for the specified destination and dates.
    Does NOT return specific hotel recommendations.
    """
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        logging.error("TAVILY_API_KEY environment variable not found.")
        return "Error: Hotel search API key (Tavily) not found."

    logging.info(f"Searching for hotel booking links via Tavily: Dest={destination}, Dates={start_date}-{end_date}")

    query = f"hotel booking websites for {destination} check-in {start_date} check-out {end_date}"

    logging.debug(f"Tavily Hotel Link Search Query: {query}")

    try:
        tavily_search = TavilySearchResults(max_results=4)
        results = tavily_search.invoke({"query": query}) 
        if not results or not isinstance(results, list):
            logging.warning("Tavily hotel link search returned no results or unexpected format.")
            return f"Relevant hotel booking site links for {destination} could not be found with Tavily search."

        links_found = []
        processed_urls = set() 
        known_sites = ["booking.com", "expedia", "google.com/travel/hotels", "hotels.com", "agoda.com", "trivago"]

        for result in results:
            if isinstance(result, dict) and result.get("url"):
                url = result.get("url")
                is_relevant = False
                for site in known_sites:
                    if site in url:
                        is_relevant = True
                        break
                if not is_relevant and "hotel" in url: # Broader check
                     is_relevant = True

                if is_relevant:
                    try:
                        domain = url.split('/')[2].replace('www.', '')
                        if domain not in processed_urls:
                            title = result.get("title", url) # Use title if available
                            links_found.append(f"- {title}: {url}")
                            processed_urls.add(domain)
                    except IndexError:
                         if url not in processed_urls: 
                             title = result.get("title", url)
                             links_found.append(f"- {title}: {url}")
                             processed_urls.add(url)


        if not links_found:
             logging.warning("Tavily results processed, but no relevant links extracted based on filters.")
             return f"Relevant hotel booking site links for {destination} could not be extracted from Tavily results."

        logging.info("Found relevant hotel booking links via Tavily.")
        return f"Some links you can use to search for hotels:\n" + "\n".join(links_found)

    except Exception as e:
        logging.error(f"Error during Tavily hotel link search: {e}", exc_info=True)
        return f"A problem occurred while searching for hotel links: {e}"


def get_tomtom_coordinates(city_name: str, api_key: str) -> Dict[str, Any]:
    """Helper function to get coordinates using TomTom Search API."""

    if not api_key:
        logging.error("API key missing when calling get_tomtom_coordinates.")
        return {"error": "TomTom API key missing."}

    logging.debug(f"Getting coordinates for '{city_name}' with TomTom Search API...")
    encoded_city = quote(city_name)
    url = f"https://api.tomtom.com/search/2/geocode/{encoded_city}.json?key={api_key}&limit=1"

    try:
        response = requests.get(url)
        logging.debug(f"TomTom Geocoding Response Code: {response.status_code}, Response: {response.text[:200]}...")
        response.raise_for_status()
        data = response.json()

        if data and data.get('results') and isinstance(data['results'], list) and len(data['results']) > 0:
            position = data['results'][0].get('position')
            if position and isinstance(position, dict) and 'lat' in position and 'lon' in position:
                 try:
                     lat = float(position['lat'])
                     lon = float(position['lon'])
                     logging.debug(f"TomTom coordinates found: Lat={lat}, Lon={lon}")
                     return {"lat": lat, "lon": lon}
                 except (ValueError, TypeError) as conv_err:
                     logging.error(f"TomTom coordinates could not be converted to numbers: {conv_err} - Data: {position}")
                     return {"error": f"Invalid coordinate format received from TomTom API."}
            else:
                 logging.warning(f"'position' or 'lat'/'lon' not found in TomTom Geocoding API response. Response: {data}")
                 return {"error": f"TomTom API couldn't find coordinate position for '{city_name}' (check detailed response format)."}
        else:
            logging.warning(f"TomTom Search API couldn't find coordinates for '{city_name}' or invalid response. Response: {data}")
            return {"error": f"TomTom API couldn't find coordinates for '{city_name}' (invalid response)."}

    except requests.exceptions.RequestException as e:
        logging.error(f"Error connecting to TomTom Search API: {e}", exc_info=True)
        return {"error": f"Map coordinates could not be retrieved (TomTom connection error): {e}"}
    except json.JSONDecodeError as e:
        logging.error(f"TomTom Search API response could not be parsed as JSON: {e}. Response Text: {response.text[:200]}...")
        return {"error": f"Map coordinates could not be retrieved (TomTom API response format error)."}
    except Exception as e:
        logging.error(f"Unexpected error while retrieving/processing TomTom coordinates: {e}", exc_info=True)
        return {"error": f"An unexpected problem occurred while retrieving map coordinates (TomTom): {e}"}

@tool
def get_tomtom_map_url(city_name: str) -> str:
    """
    A static map using TomTom Map Display API for the specified city
    creates an image URL. The TOMTOM_API_KEY environment variable is required.
    """
    api_key = os.getenv("TOMTOM_API_KEY")
    if not api_key:
        logging.error("TOMTOM_API_KEY environment variable not found.")
        return "Error: Required API key (TomTom) for creating a map not found."

    coords = get_tomtom_coordinates(city_name, api_key)
    if "error" in coords:
        logging.error(f"Coordinates for map URL could not be retrieved: {coords['error']}")
        return f"Error: Location information for map could not be retrieved ({city_name}). Reason: {coords['error']}"

    lat = coords['lat']
    lon = coords['lon']
    zoom = 11
    width = 600
    height = 400
    img_format = "png"
    map_url = f"https://api.tomtom.com/map/1/staticimage?key={api_key}&center={lon},{lat}&zoom={zoom}&width={width}&height={height}&format={img_format}"

    logging.info(f"TomTom static map URL created: {city_name}")
    return map_url