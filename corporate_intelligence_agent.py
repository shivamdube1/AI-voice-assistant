import os
import sys
import requests
from bs4 import BeautifulSoup
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END

class AgentState(TypedDict):
    company_name: str
    search_results: Optional[List[Dict[str, str]]]
    scraped_data: Optional[str]
    final_report: Optional[str]

def search_discovery(state: AgentState) -> Dict[str, Any]:
    company_name = state["company_name"]
    tavily_api_key = os.environ.get("TAVILY_API_KEY")
    if not tavily_api_key:
        raise ValueError("TAVILY_API_KEY environment variable is missing")

    url = "https://api.tavily.com/search"
    query = f"{company_name} official website, LinkedIn, recent financial business news"
    payload = {
        "api_key": tavily_api_key,
        "query": query,
        "search_depth": "advanced",
        "max_results": 5,
        "include_answer": False,
        "include_raw_content": False,
        "include_domains": [],
        "exclude_domains": []
    }

    response = requests.post(url, json=payload)
    response.raise_for_status()
    data = response.json()

    search_results = []
    for result in data.get("results", []):
        search_results.append({
            "url": result.get("url"),
            "title": result.get("title"),
            "content": result.get("content")
        })

    return {"search_results": search_results}

def deep_scraper(state: AgentState) -> Dict[str, Any]:
    search_results = state.get("search_results") or []
    scraped_text = ""

    for result in search_results:
        url = result.get("url")
        if not url:
            continue

        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, "html.parser")
                text = soup.get_text(separator=" ", strip=True)
                scraped_text += f"\n--- Content from {url} ---\n"
                scraped_text += text[:5000]
        except Exception as e:
            print(f"Failed to scrape {url}: {e}")

    return {"scraped_data": scraped_text}

def data_extraction_synthesis(state: AgentState) -> Dict[str, Any]:
    company_name = state["company_name"]
    scraped_data = state.get("scraped_data") or ""

    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable is missing")

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {openai_api_key}",
        "Content-Type": "application/json"
    }

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

    user_prompt = f"Target Company: {company_name}\n\nScraped Data:\n{scraped_data}"

    payload = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.2
    }

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()

    final_report = data["choices"][0]["message"]["content"]
    return {"final_report": final_report}

def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("search_discovery", search_discovery)
    workflow.add_node("deep_scraper", deep_scraper)
    workflow.add_node("data_extraction_synthesis", data_extraction_synthesis)

    workflow.add_edge(START, "search_discovery")
    workflow.add_edge("search_discovery", "deep_scraper")
    workflow.add_edge("deep_scraper", "data_extraction_synthesis")
    workflow.add_edge("data_extraction_synthesis", END)

    app = workflow.compile()
    return app

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python corporate_intelligence_agent.py \"[Company Name]\"")
        sys.exit(1)

    company_name = sys.argv[1]
    print(f"Running intelligence agent for: {company_name}")

    app = build_graph()
    initial_state: AgentState = {
        "company_name": company_name,
        "search_results": None,
        "scraped_data": None,
        "final_report": None
    }

    try:
        final_state = app.invoke(initial_state)
        print("\n--- Final Report ---\n")
        print(final_state.get("final_report", "No report generated."))
    except Exception as e:
        print(f"Error running agent: {e}")
