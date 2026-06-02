import argparse
import os
import requests
import json
from bs4 import BeautifulSoup
from typing import TypedDict, List
from langgraph.graph import StateGraph, END

class GraphState(TypedDict):
    company_name: str
    search_queries: List[str]
    urls_to_scrape: List[str]
    scraped_data: str
    final_report: str

def search_discovery(state: GraphState) -> GraphState:
    print("--- SEARCH & DISCOVERY NODE ---")
    company_name = state["company_name"]
    tavily_api_key = os.environ.get("TAVILY_API_KEY")
    if not tavily_api_key:
        raise ValueError("TAVILY_API_KEY environment variable is missing.")

    # We formulate queries to find official site, linkedin, and news
    queries = [
        f"{company_name} official website",
        f"{company_name} company linkedin",
        f"{company_name} recent financial business news"
    ]

    urls = []
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
                }
            )
            response.raise_for_status()
            data = response.json()
            for result in data.get("results", []):
                urls.append(result["url"])
        except Exception as e:
            print(f"Error during Tavily search for query '{query}': {e}")

    # Deduplicate URLs
    urls = list(set(urls))
    print(f"Found {len(urls)} URLs to scrape.")

    state["search_queries"] = queries
    state["urls_to_scrape"] = urls
    return state

def deep_scraper(state: GraphState) -> GraphState:
    print("--- DEEP SCRAPER NODE ---")
    urls = state.get("urls_to_scrape", [])
    all_scraped_text = []

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    for url in urls:
        print(f"Scraping: {url}")
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            # Extract text from p, h1-h6, li tags
            text_elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li'])
            text = " ".join([elem.get_text(strip=True) for elem in text_elements])

            # Basic cleanup
            text = " ".join(text.split())
            if text:
                all_scraped_text.append(f"Source: {url}\nContent: {text[:5000]}") # limit length per page
        except Exception as e:
            print(f"Failed to scrape {url}: {e}")

    state["scraped_data"] = "\n\n".join(all_scraped_text)
    return state

def data_extraction_synthesis(state: GraphState) -> GraphState:
    print("--- DATA EXTRACTION & SYNTHESIS NODE ---")
    scraped_data = state.get("scraped_data", "")
    company_name = state["company_name"]

    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable is missing.")

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

    user_prompt = f"Target Company: {company_name}\n\nHere is the scraped data:\n{scraped_data[:20000]}\n\nPlease generate the comprehensive Markdown report based on the provided instructions."

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
            }
        )
        response.raise_for_status()
        data = response.json()
        report = data["choices"][0]["message"]["content"]
        state["final_report"] = report
    except Exception as e:
        print(f"Error during OpenAI API call: {e}")
        state["final_report"] = "Error generating report."

    return state

def main():
    parser = argparse.ArgumentParser(description="Corporate Intelligence Agent")
    parser.add_argument("company_name", type=str, help="The name of the target company")
    args = parser.parse_args()

    print(f"Agent starting for company: {args.company_name}")

    # Build the graph
    workflow = StateGraph(GraphState)

    workflow.add_node("search_discovery", search_discovery)
    workflow.add_node("deep_scraper", deep_scraper)
    workflow.add_node("data_extraction_synthesis", data_extraction_synthesis)

    workflow.set_entry_point("search_discovery")
    workflow.add_edge("search_discovery", "deep_scraper")
    workflow.add_edge("deep_scraper", "data_extraction_synthesis")
    workflow.add_edge("data_extraction_synthesis", END)

    app = workflow.compile()

    inputs = {"company_name": args.company_name}
    final_state = app.invoke(inputs)

    print("\n\n====== FINAL REPORT ======\n")
    print(final_state.get("final_report", "No report generated."))
    print("\n==========================\n")

if __name__ == "__main__":
    main()
