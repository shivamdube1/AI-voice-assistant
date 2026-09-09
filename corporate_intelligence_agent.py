import os
import json
import argparse
import requests
from bs4 import BeautifulSoup
from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END

class AgentState(TypedDict):
    company_name: str
    search_urls: List[str]
    scraped_data: str
    final_report: str

def search_discovery(state: AgentState) -> AgentState:
    """Finds the company's official website, LinkedIn page, and recent news."""
    print(f"[*] Discovering information for: {state['company_name']}")

    tavily_api_key = os.environ.get("TAVILY_API_KEY")
    query = f"{state['company_name']} official website linkedin recent news"

    if not tavily_api_key:
        print("[!] TAVILY_API_KEY not found. Using mock URLs.")
        return {"search_urls": [
            "https://mock-official-site.com",
            "https://linkedin.com/company/mock",
            "https://news.ycombinator.com/mock"
        ]}

    try:
        response = requests.post(
            "https://api.tavily.com/search",
            headers={"Content-Type": "application/json"},
            json={
                "api_key": tavily_api_key,
                "query": query,
                "search_depth": "basic",
                "include_answer": False,
                "max_results": 5
            }
        )
        response.raise_for_status()
        results = response.json().get("results", [])
        urls = [res["url"] for res in results if "url" in res]
        return {"search_urls": urls}
    except Exception as e:
        print(f"[!] Error during search discovery: {e}")
        return {"search_urls": []}

def deep_scraper(state: AgentState) -> AgentState:
    """Scrapes content from the discovered URLs, skipping LinkedIn."""
    print("[*] Scraping data...")
    scraped_data_parts = []

    for url in state.get("search_urls", []):
        if "linkedin.com" in url:
            print(f"[*] Skipping LinkedIn URL to avoid blocks: {url}")
            continue

        print(f"[*] Scraping: {url}")
        try:
            # Need a user-agent to bypass some basic bot blocks
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            # Extract text, join with space, and collapse whitespace
            text = " ".join(soup.stripped_strings)

            # Limit the text size per URL to avoid blowing up context windows
            scraped_data_parts.append(f"--- Data from {url} ---\n{text[:5000]}")
        except Exception as e:
            print(f"[!] Failed to scrape {url}: {e}")

    return {"scraped_data": "\n\n".join(scraped_data_parts)}

def data_extraction_synthesis(state: AgentState) -> AgentState:
    """Uses LLM to extract metrics and generate a Markdown report."""
    print("[*] Synthesizing report...")

    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        print("[!] OPENAI_API_KEY not found. Returning mock report.")
        return {"final_report": "# Mock Report\n\nNo API key provided."}

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

    user_prompt = f"Company: {state['company_name']}\n\nScraped Data:\n{state.get('scraped_data', 'No data scraped.')}\n\nGenerate the intelligence report based ONLY on the provided scraped data."

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {openai_api_key}"
            },
            json={
                "model": "gpt-4o",  # or gpt-3.5-turbo if preferred
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.2
            }
        )
        response.raise_for_status()
        report = response.json()["choices"][0]["message"]["content"]
        return {"final_report": report}
    except Exception as e:
        print(f"[!] Error during synthesis: {e}")
        return {"final_report": f"Error generating report: {e}"}

# Compile Graph Logic
def build_agent_graph():
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
    parser = argparse.ArgumentParser(description="Corporate Intelligence Agent")
    parser.add_argument("company", type=str, help="Target company name")
    args = parser.parse_args()

    graph = build_agent_graph()
    initial_state = {"company_name": args.company}

    print(f"=========================================")
    print(f"Starting Investigation: {args.company}")
    print(f"=========================================")

    final_state = graph.invoke(initial_state)

    print(f"\n=========================================")
    print(f"FINAL REPORT")
    print(f"=========================================\n")
    print(final_state.get("final_report", "No report generated."))
