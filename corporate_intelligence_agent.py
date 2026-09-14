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

def search_discovery(state: AgentState) -> dict:
    company_name = state["company_name"]
    tavily_api_key = os.environ.get("TAVILY_API_KEY")

    if not tavily_api_key:
        print("TAVILY_API_KEY not found. Using mock URLs.")
        urls = [
            f"https://www.{company_name.lower().replace(' ', '')}.com",
            f"https://www.{company_name.lower().replace(' ', '')}.com/about",
            f"https://www.crunchbase.com/organization/{company_name.lower().replace(' ', '')}"
        ]
        return {"search_urls": urls}

    try:
        response = requests.post(
            "https://api.tavily.com/search",
            headers={"Content-Type": "application/json"},
            json={
                "api_key": tavily_api_key,
                "query": f"{company_name} official website, LinkedIn page, and recent financial/business news",
                "search_depth": "basic",
                "include_answer": False,
                "max_results": 5,
            }
        )
        response.raise_for_status()
        data = response.json()
        urls = [result["url"] for result in data.get("results", [])]
        return {"search_urls": urls}
    except Exception as e:
        print(f"Error during search discovery: {e}")
        return {"search_urls": []}

def deep_scraper(state: AgentState) -> dict:
    search_urls = state.get("search_urls", [])
    scraped_data = []

    for url in search_urls:
        if "linkedin.com" in url:
            print(f"Skipping LinkedIn URL: {url}")
            continue

        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            text = soup.get_text(separator=' ', strip=True)
            scraped_data.append(f"Source: {url}\nContent: {text[:2000]}") # Limiting content length for brevity
        except Exception as e:
            print(f"Error scraping {url}: {e}")

    return {"scraped_data": "\n\n".join(scraped_data)}

def data_extraction_synthesis(state: AgentState) -> dict:
    company_name = state["company_name"]
    scraped_data = state.get("scraped_data", "")
    openai_api_key = os.environ.get("OPENAI_API_KEY")

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

    user_prompt = f"Target Company: {company_name}\n\nScraped Data:\n{scraped_data}\n\nGenerate the structured intelligence report based ONLY on the provided scraped data."

    if not openai_api_key:
        print("OPENAI_API_KEY not found. Returning a mock report.")
        mock_report = f"""# Corporate Intelligence Report: {company_name}

## Executive Summary
Mock executive summary for {company_name}.

## Company Profile
**Specialization:** Insufficient data found for this metric
**Company Size:** Insufficient data found for this metric
**Headquarters / Key Locations:** Insufficient data found for this metric

## Company Performance
**Financials / Growth:** Insufficient data found for this metric
**Market Position:** Insufficient data found for this metric
**Core Products & Offerings:** Insufficient data found for this metric
**Recent Developments:** Insufficient data found for this metric
"""
        return {"final_report": mock_report}

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {openai_api_key}"
            },
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.2
            }
        )
        response.raise_for_status()
        data = response.json()
        report = data["choices"][0]["message"]["content"]
        return {"final_report": report}
    except Exception as e:
        print(f"Error during data extraction and synthesis: {e}")
        return {"final_report": "Error generating report."}

# Build the Graph
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("search_discovery", search_discovery)
workflow.add_node("deep_scraper", deep_scraper)
workflow.add_node("data_extraction_synthesis", data_extraction_synthesis)

# Add Edges
workflow.set_entry_point("search_discovery")
workflow.add_edge("search_discovery", "deep_scraper")
workflow.add_edge("deep_scraper", "data_extraction_synthesis")
workflow.add_edge("data_extraction_synthesis", END)

# Compile
app = workflow.compile()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python corporate_intelligence_agent.py \"[Company Name]\"")
        sys.exit(1)

    company = sys.argv[1]
    print(f"Starting research on: {company}\n")

    final_state = None
    for step_output in app.stream({"company_name": company}):
        # Capture the state at each step; the last one will be our final state
        for node_name, state in step_output.items():
            print(f"Completed node: {node_name}")
            final_state = state

    print("\n" + "="*50 + "\n")
    if final_state and "final_report" in final_state:
        print(final_state["final_report"])
    else:
        print("Failed to generate report.")
