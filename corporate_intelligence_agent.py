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


def search_discovery(state: AgentState) -> AgentState:
    company_name = state["company_name"]
    query = f"{company_name} official website OR LinkedIn OR recent financial business news"

    tavily_api_key = os.environ.get("TAVILY_API_KEY")
    if not tavily_api_key:
        print("Warning: TAVILY_API_KEY not set. Using dummy URLs for testing.")
        return {"search_urls": [f"https://example.com/{company_name.replace(' ', '').lower()}"]}

    url = "https://api.tavily.com/search"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "api_key": tavily_api_key,
        "query": query,
        "search_depth": "basic",
        "include_answer": False,
        "max_results": 5
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        results = response.json().get("results", [])
        urls = [result["url"] for result in results]
    except Exception as e:
        print(f"Error during Tavily search: {e}")
        urls = []

    return {"search_urls": urls}


def deep_scraper(state: AgentState) -> AgentState:
    urls = state.get("search_urls", [])
    all_text = ""

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    for url in urls:
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract text from p, h1, h2, h3, h4, h5, h6, li
            for tags in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li']):
                text = tags.get_text(separator=' ', strip=True)
                if text:
                    all_text += text + "\n"
        except Exception as e:
            print(f"Error scraping {url}: {e}")

    # Keep it reasonably bounded
    return {"scraped_data": all_text[:20000]}


def data_extraction_synthesis(state: AgentState) -> AgentState:
    company_name = state["company_name"]
    scraped_data = state.get("scraped_data", "")

    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        print("Warning: OPENAI_API_KEY not set. Returning dummy report.")
        return {"final_report": f"# Executive Summary\nInsufficient data found for this metric."}

    system_prompt = """You are an elite Corporate Intelligence Researcher. Your mission is to investigate a target company provided by the user, scrape relevant web data, and produce a highly structured, data-driven intelligence report.

Required Report Structure:

Executive Summary: A concise, one-paragraph overview of the company.

Company Profile:
- Specialization: What is their exact niche, core technology, or primary service?
- Company Size: Number of employees (provide an exact number or estimated range based on scraped data).
- Headquarters / Key Locations: Primary operational bases.

Company Performance:
- Financials / Growth: Estimated revenue, funding rounds, market share, or notable growth metrics.
- Market Position: Who are their primary competitors?
- Core Products & Offerings: The specific products or services they sell and their target demographic.
- Recent Developments: Key news, leadership changes, or major events from the last 6-12 months.

Constraints: Ground your entire report strictly in the data you scrape. If a specific metric (like revenue or exact employee count) cannot be found, explicitly state 'Insufficient data found for this metric' rather than guessing. Do not include unprompted packages or dependencies; rely only on the packages explicitly confirmed in the issue or trace."""

    user_prompt = f"Target Company: {company_name}\n\nScraped Data:\n{scraped_data}\n\nPlease generate the report based on the scraped data according to the system prompt instructions."

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.2
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        report = response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error during OpenAI synthesis: {e}")
        report = "Error generating report."

    return {"final_report": report}


def compile_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("search_discovery", search_discovery)
    workflow.add_node("deep_scraper", deep_scraper)
    workflow.add_node("data_extraction_synthesis", data_extraction_synthesis)

    workflow.set_entry_point("search_discovery")
    workflow.add_edge("search_discovery", "deep_scraper")
    workflow.add_edge("deep_scraper", "data_extraction_synthesis")
    workflow.add_edge("data_extraction_synthesis", END)

    return workflow.compile()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python corporate_intelligence_agent.py \"[Company Name]\"")
        sys.exit(1)

    company = sys.argv[1]
    print(f"Starting intelligence gathering for: {company}...")

    graph = compile_graph()

    initial_state = {
        "company_name": company,
        "search_urls": [],
        "scraped_data": "",
        "final_report": ""
    }

    result = graph.invoke(initial_state)
    print("\n--- Final Report ---\n")
    print(result.get("final_report", "No report generated."))
