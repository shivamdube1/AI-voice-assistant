import os
import json
import requests
import argparse
from typing import TypedDict, List
from bs4 import BeautifulSoup
from langgraph.graph import StateGraph, START, END

class AgentState(TypedDict):
    company_name: str
    search_urls: List[str]
    scraped_data: str
    final_report: str


def search_discovery(state: AgentState) -> AgentState:
    company_name = state.get("company_name", "")
    tavily_api_key = os.environ.get("TAVILY_API_KEY")

    if not tavily_api_key:
        print("TAVILY_API_KEY not found. Using mock URLs.")
        return {"search_urls": ["https://mock-company-website.com", "https://mock-news-site.com/company"]}

    print(f"Searching for information on: {company_name}")
    query = f"{company_name} official website, LinkedIn, and recent financial/business news"

    url = "https://api.tavily.com/search"
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "api_key": tavily_api_key,
        "query": query,
        "search_depth": "basic",
        "include_answer": False,
        "include_images": False,
        "include_raw_content": False,
        "max_results": 3
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        search_urls = [result.get("url") for result in data.get("results", []) if result.get("url")]
        return {"search_urls": search_urls}
    except Exception as e:
        print(f"Error during Tavily search: {e}")
        return {"search_urls": []}


def deep_scraper(state: AgentState) -> AgentState:
    search_urls = state.get("search_urls", [])
    scraped_texts = []

    print(f"Scraping {len(search_urls)} URLs...")
    for url in search_urls:
        if "linkedin.com" in url:
            print(f"Skipping linkedin.com URL to avoid blocks: {url}")
            continue

        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')
            # Extract text from p, h1-h6, li, etc to get meaningful content and avoid too much noise
            text_elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li'])
            page_text = " ".join([element.get_text(strip=True) for element in text_elements])

            # Keep it reasonable length per page to avoid blowing up context window
            page_text = page_text[:5000]

            if page_text:
                scraped_texts.append(f"--- Data from {url} ---\n{page_text}")
        except Exception as e:
            print(f"Failed to scrape {url}: {e}")

    combined_data = "\n\n".join(scraped_texts)

    if not combined_data:
        combined_data = "No data could be extracted from the provided URLs."

    return {"scraped_data": combined_data}


def data_extraction_synthesis(state: AgentState) -> AgentState:
    company_name = state.get("company_name", "")
    scraped_data = state.get("scraped_data", "")
    openai_api_key = os.environ.get("OPENAI_API_KEY")

    if not openai_api_key:
        print("OPENAI_API_KEY not found. Returning mock report.")
        return {"final_report": f"# Executive Summary\nMock report for {company_name}.\n\n# Company Profile\nInsufficient data found for this metric."}

    print(f"Synthesizing report for {company_name}...")

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

    user_prompt = f"Target Company: {company_name}\n\nScraped Data:\n{scraped_data}\n\nPlease generate the report based on this data."

    url = "https://api.openai.com/v1/chat/completions"
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
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        report = data["choices"][0]["message"]["content"]
        return {"final_report": report}
    except Exception as e:
        print(f"Error during OpenAI synthesis: {e}")
        return {"final_report": "Error generating report."}


def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("search_discovery", search_discovery)
    workflow.add_node("deep_scraper", deep_scraper)
    workflow.add_node("data_extraction_synthesis", data_extraction_synthesis)

    workflow.add_edge(START, "search_discovery")
    workflow.add_edge("search_discovery", "deep_scraper")
    workflow.add_edge("deep_scraper", "data_extraction_synthesis")
    workflow.add_edge("data_extraction_synthesis", END)

    app = workflow.compile()
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autonomous Corporate Intelligence Agent")
    parser.add_argument("company_name", type=str, help="The target company's name")
    args = parser.parse_args()

    app = build_graph()

    initial_state = {"company_name": args.company_name}

    print(f"Starting analysis for: {args.company_name}...\n")

    try:
        final_state = app.invoke(initial_state)
        print("\n=== FINAL REPORT ===\n")
        print(final_state.get("final_report", "No report generated."))
    except Exception as e:
        print(f"Error running agent pipeline: {e}")
