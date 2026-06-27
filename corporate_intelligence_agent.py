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
    scraped_data: str
    final_report: str

def search_discovery(state: AgentState):
    company_name = state["company_name"]
    tavily_api_key = os.environ.get("TAVILY_API_KEY", "")

    query = f"{company_name} official website OR {company_name} linkedin OR {company_name} recent financial business news"

    url = "https://api.tavily.com/search"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "api_key": tavily_api_key,
        "query": query,
        "max_results": 5
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        results = response.json().get("results", [])
        search_urls = [res.get("url") for res in results if "url" in res]
    except Exception as e:
        print(f"Error in search_discovery: {e}")
        search_urls = []

    return {"search_urls": search_urls}

def deep_scraper(state: AgentState):
    search_urls = state.get("search_urls", [])
    scraped_data_parts = []

    for url in search_urls:
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract text
            text = soup.get_text(separator=' ', strip=True)

            # Simple summarization or truncation to avoid blowing up context window
            scraped_data_parts.append(f"Source: {url}\nContent: {text[:2000]}")
        except Exception as e:
            print(f"Error scraping {url}: {e}")

    scraped_data = "\n\n".join(scraped_data_parts)
    return {"scraped_data": scraped_data}

def data_extraction_synthesis(state: AgentState):
    company_name = state["company_name"]
    scraped_data = state.get("scraped_data", "")
    openai_api_key = os.environ.get("OPENAI_API_KEY", "")

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

    user_prompt = f"Target Company: {company_name}\n\nScraped Data:\n{scraped_data}\n\nPlease generate the report based on the provided scraped data."

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }
    data = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.2
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        final_report = response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error in data_extraction_synthesis: {e}")
        final_report = "Error generating report."

    return {"final_report": final_report}

# Build Graph
workflow = StateGraph(AgentState)

workflow.add_node("search_discovery", search_discovery)
workflow.add_node("deep_scraper", deep_scraper)
workflow.add_node("data_extraction_synthesis", data_extraction_synthesis)

workflow.set_entry_point("search_discovery")
workflow.add_edge("search_discovery", "deep_scraper")
workflow.add_edge("deep_scraper", "data_extraction_synthesis")
workflow.add_edge("data_extraction_synthesis", END)

app = workflow.compile()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python corporate_intelligence_agent.py \"[Company Name]\"")
        sys.exit(1)

    company_name = sys.argv[1]

    print(f"Starting research on {company_name}...")

    inputs = {"company_name": company_name}

    # Run the graph
    result = app.invoke(inputs)

    print("\n--- Final Report ---\n")
    print(result.get("final_report", "No report generated."))
