import os
import json
import requests
from bs4 import BeautifulSoup
from typing import TypedDict, List
from langgraph.graph import StateGraph, END

class AgentState(TypedDict):
    company_name: str
    search_urls: List[str]
    scraped_data: str
    final_report: str

# API Fallback Logic setup
def get_tavily_api_key():
    return os.environ.get("TAVILY_API_KEY")

def get_openai_api_key():
    return os.environ.get("OPENAI_API_KEY")


def search_discovery(state: AgentState) -> AgentState:
    """
    Search & Discovery Node: Use a search tool (e.g., Tavily) to find the company's official website,
    LinkedIn page, and recent financial/business news.
    """
    company_name = state.get("company_name", "")
    api_key = get_tavily_api_key()

    if not api_key:
        # Fallback to mock URLs if no API key is provided
        mock_urls = [
            f"https://www.{company_name.lower().replace(' ', '')}.com",
            f"https://www.{company_name.lower().replace(' ', '')}.com/about-us",
            f"https://www.linkedin.com/company/{company_name.lower().replace(' ', '')}",
            f"https://news.ycombinator.com/item?id=mock_{company_name.lower().replace(' ', '')}"
        ]
        state["search_urls"] = mock_urls
        return state

    url = "https://api.tavily.com/search"
    headers = {"Content-Type": "application/json"}
    payload = {
        "api_key": api_key,
        "query": f"{company_name} official website, linkedin, business news",
        "search_depth": "advanced",
        "max_results": 5
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        urls = [result.get("url") for result in data.get("results", []) if result.get("url")]
        state["search_urls"] = urls
    except Exception as e:
        print(f"Error during search discovery: {e}")
        state["search_urls"] = []

    return state




def deep_scraper(state: AgentState) -> AgentState:
    """
    Deep Scraper Node: Extract text from the official website and external aggregators
    to hunt for employee headcounts, performance metrics, and core specializations.
    Skips LinkedIn URLs to avoid blocks.
    """
    urls = state.get("search_urls", [])
    scraped_texts = []

    for url in urls:
        if "linkedin.com" in url:
            print(f"Skipping {url} to avoid request blocks.")
            continue

        try:
            # We use a simple get request
            # In a real app we might use a fake user agent
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                # Extract text and remove extra whitespace
                text = soup.get_text(separator=' ', strip=True)
                # Keep a larger size per page (e.g. 5000 characters) to avoid losing critical intelligence
                scraped_texts.append(f"Source: {url}\nContent: {text[:5000]}")
            else:
                print(f"Failed to fetch {url}: Status {response.status_code}")
        except Exception as e:
            print(f"Error scraping {url}: {e}")

    state["scraped_data"] = "\n\n".join(scraped_texts)
    return state


def data_extraction_synthesis(state: AgentState) -> AgentState:
    """
    Data Extraction & Synthesis Node: Pass the raw data through an LLM to extract the targeted business metrics,
    filter out noise, and generate a final Markdown report.
    """
    api_key = get_openai_api_key()
    if not api_key:
        state["final_report"] = "Error: OPENAI_API_KEY is not set."
        return state

    company_name = state.get("company_name", "")
    scraped_data = state.get("scraped_data", "")

    if not scraped_data:
        state["final_report"] = f"# Executive Summary\nInsufficient data found for {company_name}."
        return state

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    system_prompt = """You are an elite Corporate Intelligence Researcher. Your mission is to investigate a target company provided by the user, scrape relevant web data, and produce a highly structured, data-driven intelligence report.

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

    user_prompt = f"Target Company: {company_name}\n\nScraped Data:\n{scraped_data}\n\nGenerate the structured Markdown report based ONLY on the provided scraped data."

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.2
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        report = result["choices"][0]["message"]["content"]
        state["final_report"] = report
    except Exception as e:
        print(f"Error during data synthesis: {e}")
        state["final_report"] = f"Error generating report: {e}"

    return state

def build_graph():
    """
    Constructs and returns the compiled LangGraph state machine.
    """
    builder = StateGraph(AgentState)

    # Add Nodes
    builder.add_node("search_discovery", search_discovery)
    builder.add_node("deep_scraper", deep_scraper)
    builder.add_node("data_extraction_synthesis", data_extraction_synthesis)

    # Define Edges (linear pipeline)
    builder.set_entry_point("search_discovery")
    builder.add_edge("search_discovery", "deep_scraper")
    builder.add_edge("deep_scraper", "data_extraction_synthesis")
    builder.add_edge("data_extraction_synthesis", END)

    return builder.compile()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python corporate_intelligence_agent.py \"[Company Name]\"")
        sys.exit(1)

    company_name = sys.argv[1]

    print(f"Starting research on: {company_name}...\n")

    app = build_graph()
    initial_state = {"company_name": company_name, "search_urls": [], "scraped_data": "", "final_report": ""}

    final_state = None
    for state_update in app.stream(initial_state):
        # We can extract the latest state from the current node step
        if "data_extraction_synthesis" in state_update:
            final_state = state_update["data_extraction_synthesis"]
        elif "deep_scraper" in state_update:
            final_state = state_update["deep_scraper"]
        elif "search_discovery" in state_update:
            final_state = state_update["search_discovery"]

    if final_state and final_state.get("final_report"):
        print("--- Final Report ---")
        print(final_state["final_report"])
    else:
        print("Failed to generate report.")
