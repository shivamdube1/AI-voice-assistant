import os
import sys
import json
import requests
from bs4 import BeautifulSoup
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END

class AgentState(TypedDict):
    company_name: str
    search_queries: List[str]
    urls_to_scrape: List[str]
    raw_scraped_data: Dict[str, str]
    final_report: str

def search_discovery(state: AgentState) -> AgentState:
    company_name = state["company_name"]
    tavily_api_key = os.environ.get("TAVILY_API_KEY")
    if not tavily_api_key:
        raise ValueError("TAVILY_API_KEY environment variable is not set")

    # Construct search queries
    queries = [
        f"{company_name} official website",
        f"{company_name} LinkedIn page",
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
                    "include_images": False,
                    "include_raw_content": False,
                    "max_results": 3,
                },
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            for result in data.get("results", []):
                urls.append(result.get("url"))
        except Exception as e:
            print(f"Error searching for query '{query}': {e}")

    # Remove duplicates while preserving order
    unique_urls = list(dict.fromkeys([url for url in urls if url]))

    return {
        "search_queries": queries,
        "urls_to_scrape": unique_urls
    }

def deep_scraper(state: AgentState) -> AgentState:
    urls = state.get("urls_to_scrape", [])
    raw_data = {}

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    for url in urls[:5]: # Limit to top 5 to avoid overly long prompts
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            # Remove scripts and styles
            for script in soup(["script", "style"]):
                script.extract()

            text = soup.get_text(separator=' ', strip=True)
            # Basic text compression: remove extra spaces
            compressed_text = ' '.join(text.split())
            # Truncate to reasonable length per page to avoid token limits
            raw_data[url] = compressed_text[:2000]
        except Exception as e:
            print(f"Error scraping {url}: {e}")

    return {
        "raw_scraped_data": raw_data
    }

def data_extraction_synthesis(state: AgentState) -> AgentState:
    company_name = state["company_name"]
    raw_data = state.get("raw_scraped_data", {})

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

    # Compile the raw data into a prompt
    data_context = "\n\n".join([f"Source: {url}\nContent: {text}" for url, text in raw_data.items()])

    user_prompt = f"Target Company: {company_name}\n\nHere is the raw scraped data:\n\n{data_context}\n\nPlease generate the required structured Markdown report based ONLY on this data."

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
        final_report = result["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error generating report: {e}")
        final_report = f"Failed to generate report: {e}"

    return {
        "final_report": final_report
    }

def create_graph():
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
        print("Usage: python corporate_intelligence_agent.py \"[Company Name]\"")
        sys.exit(1)

    company_name = sys.argv[1]
    print(f"Starting research on {company_name}...")

    # Check for API keys
    if not os.environ.get("TAVILY_API_KEY"):
        print("Warning: TAVILY_API_KEY environment variable is missing. Search functionality may fail.")
    if not os.environ.get("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY environment variable is missing. Report generation may fail.")

    graph = create_graph()

    try:
        # Initialize the state
        initial_state = {
            "company_name": company_name,
            "search_queries": [],
            "urls_to_scrape": [],
            "raw_scraped_data": {},
            "final_report": ""
        }

        # Run the graph
        result = graph.invoke(initial_state)

        print("\n" + "="*80)
        print("CORPORATE INTELLIGENCE REPORT")
        print("="*80 + "\n")
        print(result.get("final_report", "No report generated."))
        print("\n" + "="*80)

    except Exception as e:
        print(f"\nPipeline failed: {e}")
        sys.exit(1)
