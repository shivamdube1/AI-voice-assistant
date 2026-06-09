import os
import sys
import argparse
import requests
from bs4 import BeautifulSoup
from typing import TypedDict, List, Dict, Any

class AgentState(TypedDict):
    company_name: str
    search_queries: List[str]
    urls_to_scrape: List[str]
    scraped_data: str
    final_report: str

def get_args():
    parser = argparse.ArgumentParser(description="Corporate Intelligence Agent")
    parser.add_argument("company_name", type=str, help="The target company name")
    return parser.parse_args()

def search_discovery(state: AgentState) -> AgentState:
    print("--- SEARCH & DISCOVERY ---")
    company_name = state.get("company_name", "")
    tavily_api_key = os.environ.get("TAVILY_API_KEY")
    if not tavily_api_key:
        raise ValueError("TAVILY_API_KEY environment variable is not set")

    queries = [
        f"{company_name} official website",
        f"{company_name} linkedin company page",
        f"{company_name} recent financial business news revenue employees"
    ]

    urls_to_scrape = []

    for query in queries:
        try:
            response = requests.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": tavily_api_key,
                    "query": query,
                    "search_depth": "basic",
                    "include_answer": False,
                    "max_results": 2
                },
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            for result in data.get("results", []):
                urls_to_scrape.append(result.get("url"))
        except Exception as e:
            print(f"Error searching for '{query}': {e}")

    # Deduplicate URLs
    urls_to_scrape = list(set([url for url in urls_to_scrape if url]))

    return {
        **state,
        "search_queries": queries,
        "urls_to_scrape": urls_to_scrape
    }

def deep_scraper(state: AgentState) -> AgentState:
    print("--- DEEP SCRAPER ---")
    urls_to_scrape = state.get("urls_to_scrape", [])
    scraped_data_parts = []

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    for url in urls_to_scrape:
        print(f"Scraping: {url}")
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            # Remove scripts and styles
            for script_or_style in soup(["script", "style"]):
                script_or_style.extract()

            text = soup.get_text(separator=' ')

            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)

            # Limit text per URL to not blow up context window
            text = text[:3000]

            scraped_data_parts.append(f"URL: {url}\nCONTENT:\n{text}\n\n")
        except Exception as e:
            print(f"Failed to scrape {url}: {e}")

    return {
        **state,
        "scraped_data": "".join(scraped_data_parts)
    }

def data_extraction_synthesis(state: AgentState) -> AgentState:
    print("--- DATA EXTRACTION & SYNTHESIS ---")
    company_name = state.get("company_name", "")
    scraped_data = state.get("scraped_data", "")

    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")

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

    user_prompt = f"Target Company: {company_name}\n\nScraped Data:\n{scraped_data}\n\nPlease generate the intelligence report."

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {openai_api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "gpt-4o",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.2
            },
            timeout=60
        )
        response.raise_for_status()
        result = response.json()
        report = result["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        report = "Failed to generate report due to API error."

    return {
        **state,
        "final_report": report
    }

from langgraph.graph import StateGraph, END

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

def main():
    args = get_args()
    company_name = args.company_name

    print(f"Starting analysis for: {company_name}\n")

    app = build_graph()

    initial_state = {
        "company_name": company_name,
        "search_queries": [],
        "urls_to_scrape": [],
        "scraped_data": "",
        "final_report": ""
    }

    try:
        final_state = app.invoke(initial_state)
        print("\n=== FINAL REPORT ===\n")
        print(final_state.get("final_report", "No report generated."))
    except Exception as e:
        print(f"\nPipeline execution failed: {e}")

if __name__ == "__main__":
    main()
