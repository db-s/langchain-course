import os

from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AnyMessage
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
from tavily import TavilyClient

tavily = TavilyClient()

@tool
def search(query: str) -> str | dict:
    """
    Tool that searches over the internet
    Args:ccc
        query (str): The query to search for
    Returns:
        The search result
    """
    print(f"Search for {query}")
    return tavily.search(query=query)


# Using Anthropic model
# llm = ChatAnthropic(model=os.environ.get("ANTHROPIC_MODEL") or "claude-haiku-5-5")

# Using Ollama model (open source)
llm = ChatOllama(model=os.environ.get("OLLAMA_MODEL") or "qwen2.5:7b")

# Using custom tool function search()
tools = [search]

# Use the pre-defined tool from Tavily that queries the Tavily Search API and gets back json response
# tools = [TavilySearch()]

agent = create_agent(model=llm, tools=tools)


def main():
    print("Hello from langchain-course!")
    # search_query = "What's the weather in Tokyo?"
    search_query = "search for 3 job postings for an ai engineer using langchain in kolkata on linkedin and list their details"
    result = agent.invoke({"messages": [HumanMessage(content=search_query)]})
    print(result)


if __name__ == "__main__":
    main()
