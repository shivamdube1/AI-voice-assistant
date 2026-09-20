import os
import json
import requests
from bs4 import BeautifulSoup
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class AgentState(TypedDict):
    company_name: str
    search_urls: list[str]
    scraped_data: str
    final_report: str

def search_discovery(state: AgentState) -> dict:
    company_name = state["company_name"]
    tavily_api_key = os.environ.get("TAVILY_API_KEY")

    if not tavily_api_key:
        print("TAVILY_API_KEY not found, using mock URLs.")
        return {"search_urls": [f"https://example.com/about-{company_name.lower().replace(' ', '-')}", f"https://example.com/news-{company_name.lower().replace(' ', '-')}"]}

    query = f"{company_name} official website, LinkedIn, recent financial news"

    headers = {
        "Content-Type": "application/json"
    }

    payload = {
        "api_key": tavily_api_key,
        "query": query,
        "search_depth": "basic",
        "include_domains": [],
        "exclude_domains": [],
        "max_results": 3,
        "include_answer": False,
        "include_raw_content": False
    }

    try:
        response = requests.post("https://api.tavily.com/search", json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        urls = [result["url"] for result in data.get("results", [])]
        return {"search_urls": urls}
    except Exception as e:
        print(f"Error during search: {e}")
        return {"search_urls": []}

def deep_scraper(state: AgentState) -> dict:
    urls = state.get("search_urls", [])
    scraped_data_parts = []

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    for url in urls:
        if "linkedin.com" in url:
            print(f"Skipping linkedin.com URL to avoid request blocks: {url}")
            continue

        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # Basic text extraction, stripping out excessive whitespace
            text_content = ' '.join(soup.stripped_strings)
            # Limit text length to avoid token bloat
            text_content = text_content[:2000]
            scraped_data_parts.append(f"Data from {url}:\n{text_content}\n")
        except Exception as e:
            print(f"Error scraping {url}: {e}")

    return {"scraped_data": "\n---\n".join(scraped_data_parts)}

def data_extraction_synthesis(state: AgentState) -> dict:
    company_name = state["company_name"]
    scraped_data = state.get("scraped_data", "")
    openai_api_key = os.environ.get("OPENAI_API_KEY")

    if not openai_api_key:
        print("OPENAI_API_KEY not found, returning mock report.")
        return {"final_report": f"# Report for {company_name}\n\n**Executive Summary**\nInsufficient data found for this metric.\n"}

    system_prompt = """You are an elite Corporate Intelligence Researcher. Your mission is to investigate a target company provided by the user, scrape relevant web data, and produce a highly structured, data-driven intelligence report.

Required Report Structure:

Executive Summary: A concise, one-paragraph overview of the company.

Company Profile:
- Specialization: What is their exact niche, core technology, or primary service?
- Company Size: Number of employees (provide an exact number or estimated range based on scraped data).
- Headquarters / Key Locations: Primary operational bases.

Company Performance:
- Financials / Growth: Estimated revenue, funding rounds, market share, or notable growth metrics.
- Market Position: Who are their primary competitors?
- Core Products & Offerings: The specific products or services they sell and their target demographic.
- Recent Developments: Key news, leadership changes, or major events from the last 6-12 months.

Constraints: Ground your entire report strictly in the data you scrape. If a specific metric (like revenue or exact employee count) cannot be found, explicitly state 'Insufficient data found for this metric' rather than guessing."""

    user_prompt = f"Target Company: {company_name}\n\nScraped Data:\n{scraped_data}\n\nPlease generate the required structured report based ONLY on the provided scraped data."

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.2
    }

    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        report = data["choices"][0]["message"]["content"]
        return {"final_report": report}
    except Exception as e:
        print(f"Error during extraction and synthesis: {e}")
        return {"final_report": f"Error generating report: {e}"}

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
    import sys
    if len(sys.argv) < 2:
        print("Usage: python corporate_intelligence_agent.py \"[Company Name]\"")
        sys.exit(1)

    target_company = sys.argv[1]

    app = build_graph()

    initial_state = {"company_name": target_company}

    print(f"Running Corporate Intelligence Agent for: {target_company}...\n")

    final_state = None
    for output in app.stream(initial_state):
        for key, value in output.items():
            print(f"Finished node: {key}")
            final_state = value

    print("\n--- FINAL REPORT ---\n")
    if final_state and "final_report" in final_state:
        print(final_state["final_report"])
    else:
        print("Error: Could not retrieve final report.")
