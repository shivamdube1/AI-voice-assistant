import os
import sys
import json
import requests
from bs4 import BeautifulSoup
from typing import TypedDict, List
from langgraph.graph import StateGraph, END

class AgentState(TypedDict):
    company_name: str
    search_urls: List[str]
    raw_scraped_data: str
    final_report: str

def search_discovery(state: AgentState):
    company_name = state["company_name"]
    tavily_api_key = os.environ.get("TAVILY_API_KEY")

    if not tavily_api_key:
        print("Warning: TAVILY_API_KEY not found. Skipping search.")
        return {"search_urls": []}

    url = "https://api.tavily.com/search"
    headers = {"Content-Type": "application/json"}

    queries = [
        f"{company_name} official website",
        f"{company_name} company linkedin",
        f"{company_name} recent financial business news"
    ]

    all_urls = []

    for query in queries:
        payload = {
            "api_key": tavily_api_key,
            "query": query,
            "search_depth": "basic",
            "max_results": 3
        }
        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            for result in data.get("results", []):
                all_urls.append(result["url"])
        except Exception as e:
            print(f"Search error for query '{query}': {e}")

    # Deduplicate URLs
    unique_urls = list(set(all_urls))
    return {"search_urls": unique_urls}


def deep_scraper(state: AgentState):
    urls = state.get("search_urls", [])
    scraped_texts = []

    # We add User-Agent to avoid some basic 403s
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    for url in urls[:5]: # Limit to top 5 to avoid massive payloads
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                # Remove scripts and styles
                for script in soup(["script", "style"]):
                    script.extract()
                text = soup.get_text(separator=' ', strip=True)
                # Take snippet to prevent context limit overflow
                scraped_texts.append(f"Source: {url}\nContent: {text[:2000]}")
        except Exception as e:
            print(f"Scraping error for URL {url}: {e}")

    return {"raw_scraped_data": "\n\n---\n\n".join(scraped_texts)}

def data_extraction_synthesis(state: AgentState):
    company_name = state["company_name"]
    raw_data = state.get("raw_scraped_data", "")
    openai_api_key = os.environ.get("OPENAI_API_KEY")

    if not openai_api_key:
        return {"final_report": "Error: OPENAI_API_KEY not found."}

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
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

    user_prompt = f"Company Name: {company_name}\n\nHere is the raw scraped data. Follow the constraints and generate the report.\n\nRaw Data:\n{raw_data}"

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.2
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        report = data["choices"][0]["message"]["content"]
        return {"final_report": report}
    except Exception as e:
        return {"final_report": f"Error generating report: {e}"}


# Build Graph
def build_agent():
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

    company = sys.argv[1]
    print(f"Starting investigation for: {company}")

    app = build_agent()

    # Initialize state
    initial_state = {"company_name": company}

    # Run the graph
    result = app.invoke(initial_state)

    print("\n" + "="*50 + "\n")
    print(result.get("final_report", "No report generated."))
    print("\n" + "="*50 + "\n")
