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

def search_discovery(state: AgentState) -> AgentState:
    company_name = state["company_name"]
    tavily_api_key = os.environ.get("TAVILY_API_KEY")

    if tavily_api_key:
        query = f"{company_name} official website OR {company_name} linkedin OR {company_name} recent financial business news"
        headers = {"Content-Type": "application/json"}
        data = {
            "api_key": tavily_api_key,
            "query": query,
            "search_depth": "advanced",
            "include_domains": [],
            "exclude_domains": [],
            "max_results": 5
        }

        try:
            response = requests.post("https://api.tavily.com/search", json=data, headers=headers)
            response.raise_for_status()
            results = response.json().get("results", [])
            urls = [result["url"] for result in results]
        except Exception as e:
            print(f"Error calling Tavily API: {e}")
            urls = []
    else:
        # Default mock URLs if TAVILY_API_KEY is not set
        urls = [
            f"https://www.{company_name.lower().replace(' ', '')}.com",
            f"https://www.linkedin.com/company/{company_name.lower().replace(' ', '')}",
            f"https://news.ycombinator.com/item?id=12345" # mock news
        ]

    return {"search_urls": urls}

def deep_scraper(state: AgentState) -> AgentState:
    search_urls = state.get("search_urls", [])
    scraped_text = ""

    for url in search_urls:
        if "linkedin.com" in url.lower():
            continue  # Skip linkedin to avoid blocks

        try:
            # Set a generic user agent to help avoid basic blocks
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                # Extract text, removing scripts and styles
                for script in soup(["script", "style"]):
                    script.extract()
                text = soup.get_text(separator=' ', strip=True)
                scraped_text += f"\n--- Data from {url} ---\n"
                scraped_text += text[:5000] # Limit text per URL to avoid overwhelming LLM context
        except Exception as e:
            print(f"Error scraping {url}: {e}")

    return {"scraped_data": scraped_text}

def data_extraction_synthesis(state: AgentState) -> AgentState:
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

Constraints: Ground your entire report strictly in the data you scrape. If a specific metric (like revenue or exact employee count) cannot be found, explicitly state 'Insufficient data found for this metric' rather than guessing.
"""
    user_prompt = f"Target Company: {company_name}\n\nScraped Data:\n{scraped_data}\n\nGenerate the intelligence report based on the provided data."

    if openai_api_key:
        headers = {
            "Authorization": f"Bearer {openai_api_key}",
            "Content-Type": "application/json"
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
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
            response.raise_for_status()
            report = response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            report = "Error generating report."
    else:
        # Fallback mock report if no API key
        report = f"""# Corporate Intelligence Report: {company_name}

## Executive Summary
Insufficient data found for this metric.

## Company Profile
**Specialization:** Insufficient data found for this metric.
**Company Size:** Insufficient data found for this metric.
**Headquarters / Key Locations:** Insufficient data found for this metric.

## Company Performance
**Financials / Growth:** Insufficient data found for this metric.
**Market Position:** Insufficient data found for this metric.
**Core Products & Offerings:** Insufficient data found for this metric.
**Recent Developments:** Insufficient data found for this metric.
"""

    return {"final_report": report}

def create_agent():
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
        print('Usage: python corporate_intelligence_agent.py "[Company Name]"')
        sys.exit(1)

    company_name = sys.argv[1]

    app = create_agent()

    initial_state = AgentState(
        company_name=company_name,
        search_urls=[],
        scraped_data="",
        final_report=""
    )

    final_state = None
    for output in app.stream(initial_state):
        for key, value in output.items():
            print(f"Finished node: {key}")
            final_state = value

    if final_state and "final_report" in final_state:
        print("\n--- Final Report ---\n")
        print(final_state["final_report"])
