import os
import sys
import argparse
import requests
from bs4 import BeautifulSoup
from typing import TypedDict, List, Dict, Any
from tavily import TavilyClient
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

# Load environment variables (OPENAI_API_KEY, TAVILY_API_KEY)
load_dotenv()

# State Definition
class AgentState(TypedDict):
    company_name: str
    search_results: List[Dict[str, str]]
    scraped_data: str
    report: str

# 1. Search & Discovery Node
def search_discovery(state: AgentState) -> AgentState:
    print(f"[*] Running Search & Discovery for: {state['company_name']}")
    tavily_api_key = os.environ.get("TAVILY_API_KEY")
    if not tavily_api_key:
        raise ValueError("TAVILY_API_KEY environment variable is missing.")

    client = TavilyClient(api_key=tavily_api_key)

    # We want to find official website, LinkedIn, and recent financial/business news
    query = f"{state['company_name']} official website OR LinkedIn OR recent financial business news"

    # Use search method
    response = client.search(query=query, search_depth="advanced", max_results=5)

    search_results = []
    for result in response.get("results", []):
        search_results.append({
            "url": result.get("url"),
            "content": result.get("content", "")
        })

    return {"search_results": search_results}

# 2. Deep Scraper Node
def deep_scraper(state: AgentState) -> AgentState:
    print(f"[*] Running Deep Scraper for URLs found...")
    scraped_text = ""

    # Also include the text snippet from Tavily as a base
    for res in state.get("search_results", []):
        scraped_text += f"Source: {res['url']}\nSnippet: {res['content']}\n\n"

        # Optionally, try to scrape the URL directly if it's not a platform that blocks scraping heavily (like LinkedIn)
        url = res["url"]
        if "linkedin.com" not in url and "crunchbase.com" not in url:
            try:
                response = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    # Extract text, removing scripts and styles
                    for script in soup(["script", "style"]):
                        script.extract()
                    text = soup.get_text(separator=' ', strip=True)
                    # Limit text per page to avoid massive context
                    scraped_text += f"Scraped Content ({url}):\n{text[:2000]}\n\n"
            except Exception as e:
                print(f"Failed to scrape {url}: {e}")

    return {"scraped_data": scraped_text}

# 3. Data Extraction & Synthesis Node
def data_extraction_synthesis(state: AgentState) -> AgentState:
    print("[*] Synthesizing data into report...")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0) # gpt-4o-mini is standard and fast

    system_prompt = """You are an elite Corporate Intelligence Researcher. Your mission is to investigate a target company provided by the user, scrape relevant web data, and produce a highly structured, data-driven intelligence report.

Your Execution Loop:
Plan: Identify the search queries needed to find the company's official site, corporate profiles (like LinkedIn or Crunchbase), and recent financial press.
Search & Scrape: Deploy your tools to extract raw text from these URLs. Look specifically for quantitative data.
Synthesize: Cross-reference the data, discard marketing fluff, and extract hard facts.

Required Report Structure:
Executive Summary: A concise, one-paragraph overview of the company.

Company Profile:
Specialization: What is their exact niche, core technology, or primary service?
Company Size: Number of employees (provide an exact number or estimated range based on scraped data).
Headquarters / Key Locations: Primary operational bases.

Company Performance:
Financials / Growth: Estimated revenue, funding rounds, market share, or notable growth metrics.
Market Position: Who are their primary competitors?
Core Products & Offerings: The specific products or services they sell and their target demographic.
Recent Developments: Key news, leadership changes, or major events from the last 6-12 months.

Constraints: Ground your entire report strictly in the data you scrape. If a specific metric (like revenue or exact employee count) cannot be found, explicitly state 'Insufficient data found for this metric' rather than guessing."""

    user_prompt = f"Target Company: {state['company_name']}\n\nHere is the scraped data:\n{state['scraped_data']}\n\nPlease generate the required Markdown report."

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]

    response = llm.invoke(messages)

    return {"report": response.content}

def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("search_discovery", search_discovery)
    workflow.add_node("deep_scraper", deep_scraper)
    workflow.add_node("data_extraction_synthesis", data_extraction_synthesis)

    workflow.set_entry_point("search_discovery")
    workflow.add_edge("search_discovery", "deep_scraper")
    workflow.add_edge("deep_scraper", "data_extraction_synthesis")
    workflow.add_edge("data_extraction_synthesis", END)

    return workflow.compile()

def main():
    parser = argparse.ArgumentParser(description="Corporate Intelligence Agent")
    parser.add_argument("company_name", type=str, help="Name of the company to investigate")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is missing.")
        sys.exit(1)

    if not os.environ.get("TAVILY_API_KEY"):
        print("Error: TAVILY_API_KEY environment variable is missing.")
        sys.exit(1)

    app = build_graph()
    initial_state = {"company_name": args.company_name, "search_results": [], "scraped_data": "", "report": ""}

    result = app.invoke(initial_state)
    print("\n" + "="*50 + "\n")
    print(result.get("report", "No report generated."))
    print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main()
