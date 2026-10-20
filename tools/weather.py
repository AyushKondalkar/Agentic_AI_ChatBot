
# Import libraries
import os
import requests
from dotenv import load_dotenv
from langchain_core.tools import tool

# Load API keys from the project's .env file
load_dotenv()


# Register the weather function as a LangChain tool
@tool
def get_current_weather(city: str) -> str:
    """Get the current weather for a city.
    Input: city name, such as Pune or Mumbai.
    """

    # Read the OpenWeatherMap API key
    api_key = os.getenv("OPENWEATHERMAP_API_KEY")

    # Check whether the API key is configured
    if not api_key:
        return "Error: OPENWEATHERMAP_API_KEY is missing from .env."

    # API endpoint for current weather
    url = "https://api.openweathermap.org/data/2.5/weather"

    # Parameters sent to the weather API
    params = {
        "q": city,
        "appid": api_key,
        "units": "metric",
    }

    try:
        # Send the request to OpenWeatherMap
        response = requests.get(url, params=params, timeout=10)

        # Raise an error if the API request failed
        response.raise_for_status()

        # Convert the JSON response into a Python dictionary
        data = response.json()

        # Extract useful weather information
        temperature = data["main"]["temp"]
        feels_like = data["main"]["feels_like"]
        humidity = data["main"]["humidity"]
        description = data["weather"][0]["description"]

        # Return a readable weather report
        return (
            f"Weather in {city}:\n"
            f"Temperature: {temperature}°C\n"
            f"Feels like: {feels_like}°C\n"
            f"Humidity: {humidity}%\n"
            f"Conditions: {description}"
        )

    # Handle API and connection errors
    except requests.exceptions.HTTPError as error:
        if response.status_code == 404:
            return f"Error: City '{city}' was not found."
        if response.status_code == 401:
            return "Error: Invalid API key or API key not activated."
        return f"Weather API error: {error}"

    except requests.exceptions.RequestException as error:
        return f"Error connecting to weather service: {error}"

    except (KeyError, IndexError, ValueError):
        return "Error: Could not read the weather information."
