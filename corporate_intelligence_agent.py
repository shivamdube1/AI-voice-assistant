import os
import sys
import requests
from bs4 import BeautifulSoup
from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END

class AgentState(TypedDict, total=False):
    company_name: str
    urls_to_scrape: List[str]
    scraped_text: str
    report: str

def search_discovery(state: AgentState) -> AgentState:
    company_name = state["company_name"]
    tavily_api_key = os.environ.get("TAVILY_API_KEY")
    if not tavily_api_key:
        raise ValueError("TAVILY_API_KEY environment variable is not set.")

    urls = []

    queries = [
        f"{company_name} official website",
        f"{company_name} LinkedIn page",
        f"{company_name} recent financial business news"
    ]

    for query in queries:
        payload = {
            "api_key": tavily_api_key,
            "query": query,
            "search_depth": "basic",
            "max_results": 2
        }
        response = requests.post("https://api.tavily.com/search", json=payload)
        response.raise_for_status()
        data = response.json()
        for result in data.get("results", []):
            url = result.get("url")
            if url and url not in urls:
                urls.append(url)

    return {"urls_to_scrape": urls}

def deep_scraper(state: AgentState) -> AgentState:
    urls = state.get("urls_to_scrape", [])
    scraped_texts = []

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    for url in urls:
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, 'html.parser')
                text = soup.get_text(separator=' ', strip=True)
                # Take the first 10,000 characters to keep context size manageable
                scraped_texts.append(f"URL: {url}\nContent: {text[:10000]}")
        except Exception as e:
            print(f"Failed to scrape {url}: {e}")

    combined_text = "\n\n".join(scraped_texts)
    return {"scraped_text": combined_text}

def data_extraction_synthesis(state: AgentState) -> AgentState:
    company_name = state["company_name"]
    scraped_text = state.get("scraped_text", "")
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")

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

    user_prompt = f"Target Company: {company_name}\n\nScraped Data:\n{scraped_text}"

    payload = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.2
    }

    headers = {
        "Authorization": f"Bearer {openai_api_key}",
        "Content-Type": "application/json"
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers)
    response.raise_for_status()
    data = response.json()

    report = data["choices"][0]["message"]["content"]
    return {"report": report}

def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("search_discovery", search_discovery)
    builder.add_node("deep_scraper", deep_scraper)
    builder.add_node("data_extraction_synthesis", data_extraction_synthesis)

    builder.add_edge(START, "search_discovery")
    builder.add_edge("search_discovery", "deep_scraper")
    builder.add_edge("deep_scraper", "data_extraction_synthesis")
    builder.add_edge("data_extraction_synthesis", END)

    return builder.compile()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python corporate_intelligence_agent.py "[Company Name]"')
        sys.exit(1)

    target_company = sys.argv[1]

    graph = build_graph()

    initial_state = {"company_name": target_company}

    result = graph.invoke(initial_state)

    print(result.get("report", "No report generated."))
