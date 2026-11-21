import os
from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
from tavily import TavilyClient


class Source(BaseModel):
    """Schema for a source used by the agent"""

    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """Schema for the agent response with answer and sources"""

    answer: str = Field(description="The agent's answer to the query")
    sources: List[Source] = Field(
        default_factory=list,
        description="A list of sources used to generate the answers",
    )


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
llm = ChatAnthropic(model=os.environ.get("ANTHROPIC_MODEL") or "claude-haiku-5-5")

# Using Ollama model (open source)
# llm = ChatOllama(model=os.environ.get("OLLAMA_MODEL") or "qwen2.5:7b")

# Using custom tool function search()
# tools = [search]

# Use the pre-defined tool from tavily that queries the tavily Search API and gets back json response
tools = [TavilySearch()]

agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


def main():
    print("Hello from langchain-course!")
    # search_query = "What's the weather in Tokyo?"
    search_query = "search for 3 job postings for an ai engineer using langchain in kolkata area on linkedin and list their details"
    result = agent.invoke({"messages": [HumanMessage(content=search_query)]})
    print(result)


if __name__ == "__main__":
    main()
