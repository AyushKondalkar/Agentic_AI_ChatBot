
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_tavily import TavilySearch

# Load .env from the project root
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# Check that the key exists
if not os.getenv("TAVILY_API_KEY"):
    raise ValueError("TAVILY_API_KEY is missing from the project .env file.")

# Initialize the web search tool
web_search_tool = TavilySearch(max_results=3)
