import os
import sys
import requests
from bs4 import BeautifulSoup
from typing import TypedDict, List
from langgraph.graph import StateGraph, END

class AgentState(TypedDict):
    company_name: str
    search_urls: List[str]
    scraped_data: str
    report: str

def search_discovery(state: AgentState) -> AgentState:
    tavily_api_key = os.environ.get("TAVILY_API_KEY")
    if not tavily_api_key:
        raise ValueError("TAVILY_API_KEY environment variable not set")

    company_name = state["company_name"]
    # The prompt suggests finding official website, LinkedIn page, and recent financial/business news.
    query = f"{company_name} official website OR LinkedIn OR recent financial business news"

    response = requests.post(
        "https://api.tavily.com/search",
        json={
            "api_key": tavily_api_key,
            "query": query,
            "include_domains": [],
            "exclude_domains": [],
            "max_results": 7
        },
        headers={"Content-Type": "application/json"}
    )
    response.raise_for_status()
    results = response.json().get("results", [])
    urls = [result["url"] for result in results]

    return {"search_urls": urls}

def deep_scraper(state: AgentState) -> AgentState:
    urls = state.get("search_urls", [])
    all_text = ""
    for url in urls:
        try:
            # We'll use a basic GET request. In a production app, we might need a more sophisticated scraper.
            res = requests.get(url, timeout=10)
            if res.status_code == 200:
                soup = BeautifulSoup(res.text, 'html.parser')
                text = soup.get_text(separator=' ', strip=True)
                # To prevent overloading the LLM, limit text length per url
                all_text += f"\n\n--- Content from {url} ---\n" + text[:8000]
        except Exception as e:
            all_text += f"\n\n--- Failed to scrape {url}: {e} ---\n"

    return {"scraped_data": all_text}

def data_extraction_synthesis(state: AgentState) -> AgentState:
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

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

    user_message = f"Company Name: {state['company_name']}\n\nScraped Data:\n{state['scraped_data']}"

    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            "temperature": 0.2
        },
        headers={
            "Authorization": f"Bearer {openai_api_key}",
            "Content-Type": "application/json"
        }
    )
    response.raise_for_status()
    report = response.json()["choices"][0]["message"]["content"]

    return {"report": report}

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

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python corporate_intelligence_agent.py \"[Company Name]\"")
        sys.exit(1)

    company_name = sys.argv[1]

    graph = build_graph()

    initial_state = {"company_name": company_name, "search_urls": [], "scraped_data": "", "report": ""}

    print(f"Starting analysis for: {company_name}...")
    final_state = graph.invoke(initial_state)

    print("\n" + "="*50 + "\n")
    print(final_state["report"])
    print("\n" + "="*50 + "\n")
