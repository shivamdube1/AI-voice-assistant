import os
import sys
import json
import requests
from bs4 import BeautifulSoup
from typing import TypedDict, List
from langgraph.graph import StateGraph, END

class AgentState(TypedDict):
    company_name: str
    search_queries: List[str]
    urls_to_scrape: List[str]
    scraped_data: str
    report: str

SYSTEM_PROMPT = """You are an elite Corporate Intelligence Researcher. Your mission is to investigate a target company provided by the user, scrape relevant web data, and produce a highly structured, data-driven intelligence report.

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

def search_discovery(state: AgentState) -> AgentState:
    print("--- NODE: Search & Discovery ---")
    company_name = state.get("company_name", "")
    tavily_api_key = os.environ.get("TAVILY_API_KEY")

    if not tavily_api_key:
        print("Warning: TAVILY_API_KEY not found. Returning empty search results.")
        return {"urls_to_scrape": []}

    queries = [
        f"{company_name} official website",
        f"{company_name} LinkedIn profile",
        f"{company_name} recent financial business news"
    ]

    urls = []
    headers = {
        "Content-Type": "application/json"
    }

    for query in queries:
        payload = {
            "api_key": tavily_api_key,
            "query": query,
            "search_depth": "basic",
            "include_answer": False,
            "include_images": False,
            "include_raw_content": False,
            "max_results": 2
        }
        try:
            response = requests.post("https://api.tavily.com/search", json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            for result in data.get("results", []):
                urls.append(result.get("url"))
        except Exception as e:
            print(f"Error querying Tavily for '{query}': {e}")

    # Remove duplicates
    urls = list(set([u for u in urls if u]))
    print(f"Found URLs: {urls}")
    return {"urls_to_scrape": urls, "search_queries": queries}

def deep_scraper(state: AgentState) -> AgentState:
    print("--- NODE: Deep Scraper ---")
    urls = state.get("urls_to_scrape", [])
    scraped_texts = []

    for url in urls:
        print(f"Scraping: {url}")
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            # Extract text
            text = soup.get_text(separator=' ', strip=True)
            # Take a snippet to avoid huge payloads
            scraped_texts.append(f"Source: {url}\nContent: {text[:2000]}")
        except Exception as e:
            print(f"Error scraping {url}: {e}")

    combined_data = "\n\n".join(scraped_texts)
    if not combined_data:
        combined_data = "No data could be scraped."

    return {"scraped_data": combined_data}

def data_extraction_synthesis(state: AgentState) -> AgentState:
    print("--- NODE: Data Extraction & Synthesis ---")
    scraped_data = state.get("scraped_data", "")
    company_name = state.get("company_name", "")
    openai_api_key = os.environ.get("OPENAI_API_KEY")

    if not openai_api_key:
        print("Warning: OPENAI_API_KEY not found. Returning dummy report.")
        return {"report": "API key missing. Cannot generate report."}

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }

    user_prompt = f"Target Company: {company_name}\n\nScraped Data:\n{scraped_data}\n\nPlease generate the report based on the provided scraped data."

    payload = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.2
    }

    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        report = data["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error generating report: {e}")
        report = f"Failed to generate report: {e}"

    return {"report": report}

# Compile Graph
graph_builder = StateGraph(AgentState)
graph_builder.add_node("search_discovery", search_discovery)
graph_builder.add_node("deep_scraper", deep_scraper)
graph_builder.add_node("data_extraction_synthesis", data_extraction_synthesis)

graph_builder.set_entry_point("search_discovery")
graph_builder.add_edge("search_discovery", "deep_scraper")
graph_builder.add_edge("deep_scraper", "data_extraction_synthesis")
graph_builder.add_edge("data_extraction_synthesis", END)

app = graph_builder.compile()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python corporate_intelligence_agent.py \"[Company Name]\"")
        sys.exit(1)

    target_company = sys.argv[1]
    print(f"Starting investigation on: {target_company}")

    initial_state = {
        "company_name": target_company,
        "search_queries": [],
        "urls_to_scrape": [],
        "scraped_data": "",
        "report": ""
    }

    result = app.invoke(initial_state)

    print("\n" + "="*50)
    print("FINAL INTELLIGENCE REPORT:")
    print("="*50 + "\n")
    print(result.get("report", "No report generated."))
