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
    tavily_api_key = os.environ.get("TAVILY_API_KEY")

    if not tavily_api_key:
        print(f"Warning: TAVILY_API_KEY not found. Using mock URLs for {company_name}.")
        return {"search_urls": [
            f"https://www.{company_name.lower().replace(' ', '')}.com",
            f"https://www.linkedin.com/company/{company_name.lower().replace(' ', '')}",
            f"https://www.crunchbase.com/organization/{company_name.lower().replace(' ', '')}"
        ]}

    query = f"official website, LinkedIn page, and recent financial business news for {company_name}"

    try:
        response = requests.post(
            "https://api.tavily.com/search",
            headers={"Content-Type": "application/json"},
            json={
                "api_key": tavily_api_key,
                "query": query,
                "search_depth": "basic",
                "include_answer": False,
                "include_images": False,
                "include_raw_content": False,
                "max_results": 5
            },
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        urls = [result.get("url") for result in data.get("results", []) if result.get("url")]
        return {"search_urls": urls}
    except Exception as e:
        print(f"Error during search: {e}")
        return {"search_urls": []}


def deep_scraper(state: AgentState) -> AgentState:
    urls = state.get("search_urls", [])
    scraped_texts = []

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    for url in urls:
        if "linkedin.com" in url.lower():
            print(f"Skipping LinkedIn URL to avoid blocks: {url}")
            continue

        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()

            text = soup.get_text(separator=' ', strip=True)
            # Basic text chunking / limitation to avoid blowing up context window
            if text:
                scraped_texts.append(f"Source: {url}\nContent: {text[:2000]}") # Taking first 2000 chars per source
        except Exception as e:
            print(f"Failed to scrape {url}: {e}")

    combined_data = "\n\n".join(scraped_texts)
    if not combined_data:
        combined_data = "No data could be scraped from the provided URLs."

    return {"scraped_data": combined_data}


def data_extraction_synthesis(state: AgentState) -> AgentState:
    company_name = state["company_name"]
    scraped_data = state["scraped_data"]
    openai_api_key = os.environ.get("OPENAI_API_KEY")

    if not openai_api_key:
        print(f"Warning: OPENAI_API_KEY not found. Returning a mock report for {company_name}.")
        mock_report = f"""# Corporate Intelligence Report: {company_name}

## Executive Summary
Insufficient data found for this metric.

## Company Profile
- **Specialization:** Insufficient data found for this metric.
- **Company Size:** Insufficient data found for this metric.
- **Headquarters / Key Locations:** Insufficient data found for this metric.

## Company Performance
- **Financials / Growth:** Insufficient data found for this metric.
- **Market Position:** Insufficient data found for this metric.
- **Core Products & Offerings:** Insufficient data found for this metric.
- **Recent Developments:** Insufficient data found for this metric.
"""
        return {"final_report": mock_report}

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

    user_prompt = f"Target Company: {company_name}\n\nScraped Data:\n{scraped_data}\n\nPlease generate the required Markdown report."

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
            },
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        report = data["choices"][0]["message"]["content"]
        return {"final_report": report}
    except Exception as e:
        print(f"Error during synthesis: {e}")
        return {"final_report": "Error generating report."}


def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("search_discovery", search_discovery)
    builder.add_node("deep_scraper", deep_scraper)
    builder.add_node("data_extraction_synthesis", data_extraction_synthesis)

    builder.set_entry_point("search_discovery")
    builder.add_edge("search_discovery", "deep_scraper")
    builder.add_edge("deep_scraper", "data_extraction_synthesis")
    builder.add_edge("data_extraction_synthesis", END)

    return builder.compile()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        company_name = sys.argv[1]
        print(f"Starting analysis for: {company_name}")
        graph = build_graph()
        result = graph.invoke({"company_name": company_name})
        print("\n\n--- Final Report ---\n\n")
        print(result.get("final_report", "No report generated."))
    else:
        print('Usage: python corporate_intelligence_agent.py "[Company Name]"')
