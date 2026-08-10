from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from dotenv import load_dotenv
import os
import requests

load_dotenv()

search_tool = DuckDuckGoSearchRun()

@tool
def get_weather_data(city: str) -> str:
    """
    Get the current weather data for a given city using OpenWeather.
    """

    api_key = os.getenv("OPENWEATHER_API_KEY")

    url = "https://api.openweathermap.org/data/2.5/weather"

    params = {
        "q": city,
        "appid": api_key,
        "units": "metric"
    }

    response = requests.get(url, params=params, timeout=10)

    if response.status_code != 200:
        return f"Could not get weather data for {city}: {response.text}"

    data = response.json()

    temperature = data["main"]["temp"]
    feels_like = data["main"]["feels_like"]
    humidity = data["main"]["humidity"]
    description = data["weather"][0]["description"]

    return (
        f"City: {data['name']}\n"
        f"Temperature: {temperature}°C\n"
        f"Feels like: {feels_like}°C\n"
        f"Humidity: {humidity}%\n"
        f"Condition: {description}"
    )

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0
)

agent = create_agent(
    model=llm,
    tools=[
        search_tool,
        get_weather_data
    ],
    system_prompt="""
You are a helpful assistant with access to two tools:

1. DuckDuckGo search:
   Use it when the user asks for information that requires web search
   or current/general factual information.

2. Weather tool:
   Use it when the user asks for current weather or temperature
   of a city.

Choose the appropriate tool based on the user's question.

If a question requires multiple pieces of information, you may use
multiple tools in sequence.

After getting the required information, give the user a clear and
concise final answer.
"""
)

# query = "What is the current temp of gurgaon"
query = "Identify the birth place city of Kalpana Chawla (search) and give it's current temperature."

response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": query
            }
        ]
    }
)

print(response["messages"][-1].content)