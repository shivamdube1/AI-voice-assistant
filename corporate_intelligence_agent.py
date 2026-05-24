import os
import argparse
import requests
from bs4 import BeautifulSoup
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults

# 1. State Definition
class AgentState(TypedDict):
    company_name: str
    search_urls: List[str]
    scraped_data: str
    final_report: str

# 2. Node Functions

def search_discovery_node(state: AgentState):
    """
    Use a search tool (Tavily) to find the company's official website,
    LinkedIn page, and recent financial/business news.
    """
    company_name = state["company_name"]
    print(f"--- SEARCH & DISCOVERY: {company_name} ---")

    # Initialize Tavily Search Tool
    # Ensure TAVILY_API_KEY is in the environment
    search_tool = TavilySearchResults(max_results=5)

    queries = [
        f"{company_name} official website",
        f"{company_name} LinkedIn",
        f"{company_name} recent financial business news"
    ]

    all_urls = []
    for query in queries:
        try:
            results = search_tool.invoke({"query": query})
            for res in results:
                if 'url' in res:
                    all_urls.append(res['url'])
        except Exception as e:
            print(f"Error during search for '{query}': {e}")

    # Remove duplicates while preserving order
    unique_urls = list(dict.fromkeys(all_urls))

    return {"search_urls": unique_urls}


def deep_scraper_node(state: AgentState):
    """
    Extract text from the official website (About Us, Careers) and external aggregators
    to specifically hunt for employee headcounts, performance metrics (revenue, growth),
    and core specializations.
    """
    print("--- DEEP SCRAPER ---")
    urls = state.get("search_urls", [])
    scraped_text_parts = []

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    # Only scrape the first few to avoid taking too long/hitting limits
    for url in urls[:5]:
        print(f"Scraping: {url}")
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                # Extract text
                text = soup.get_text(separator=' ', strip=True)
                # Keep it manageable
                scraped_text_parts.append(f"Source: {url}\nContent: {text[:2000]}")
        except Exception as e:
            print(f"Failed to scrape {url}: {e}")

    combined_text = "\n\n".join(scraped_text_parts)
    return {"scraped_data": combined_text}


def data_extraction_synthesis_node(state: AgentState):
    """
    Pass the raw data through an LLM to extract the targeted business metrics,
    filter out noise, and generate a final Markdown report.
    """
    print("--- DATA EXTRACTION & SYNTHESIS ---")
    company_name = state["company_name"]
    scraped_data = state.get("scraped_data", "")

    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    system_prompt = """You are an elite Corporate Intelligence Researcher. Your mission is to investigate a target company provided by the user, scrape relevant web data, and produce a highly structured, data-driven intelligence report.

Your Execution Loop:
Plan: Identify the search queries needed to find the company's official site, corporate profiles (like LinkedIn or Crunchbase), and recent financial press.
Search & Scrape: Deploy your tools to extract raw text from these URLs. Look specifically for quantitative data.
Synthesize: Cross-reference the data, discard marketing fluff, and extract hard facts.

Required Report Structure:

# Corporate Intelligence Report: {company_name}

## Executive Summary
A concise, one-paragraph overview of the company.

## Company Profile
- **Specialization:** What is their exact niche, core technology, or primary service?
- **Company Size:** Number of employees (provide an exact number or estimated range based on scraped data).
- **Headquarters / Key Locations:** Primary operational bases.

## Company Performance
- **Financials / Growth:** Estimated revenue, funding rounds, market share, or notable growth metrics.
- **Market Position:** Who are their primary competitors?
- **Core Products & Offerings:** The specific products or services they sell and their target demographic.

## Recent Developments
Key news, leadership changes, or major events from the last 6-12 months.

Constraints: Ground your entire report strictly in the data you scrape. If a specific metric (like revenue or exact employee count) cannot be found, explicitly state 'Insufficient data found for this metric' rather than guessing.
"""

    user_prompt = f"Target Company: {company_name}\n\nHere is the scraped raw data to base your report on:\n\n<scraped_data>\n{scraped_data}\n</scraped_data>\n\nPlease generate the required structured Markdown report."

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", user_prompt)
    ])

    chain = prompt | llm

    response = chain.invoke({"company_name": company_name})

    return {"final_report": response.content}


# 3. Build and Compile the Graph
def build_agent_graph():
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("search_discovery", search_discovery_node)
    workflow.add_node("deep_scraper", deep_scraper_node)
    workflow.add_node("data_extraction_synthesis", data_extraction_synthesis_node)

    # Define edges
    workflow.set_entry_point("search_discovery")
    workflow.add_edge("search_discovery", "deep_scraper")
    workflow.add_edge("deep_scraper", "data_extraction_synthesis")
    workflow.add_edge("data_extraction_synthesis", END)

    # Compile the graph
    app = workflow.compile()
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Corporate Intelligence Agent")
    parser.add_argument("company_name", type=str, help="The target company name")
    args = parser.parse_args()

    # Verify API keys
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is not set.")
        exit(1)
    if not os.environ.get("TAVILY_API_KEY"):
        print("Error: TAVILY_API_KEY environment variable is not set.")
        exit(1)

    app = build_agent_graph()

    initial_state = {"company_name": args.company_name}
    print(f"Starting pipeline for: {args.company_name}\n")

    try:
        final_state = app.invoke(initial_state)
        print("\n" + "="*50 + "\n")
        print(final_state["final_report"])
        print("\n" + "="*50 + "\n")
    except Exception as e:
        print(f"An error occurred during execution: {e}")
