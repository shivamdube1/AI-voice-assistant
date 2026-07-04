import os
import sys
import json
import requests
from bs4 import BeautifulSoup
from typing import TypedDict, List
from langgraph.graph import StateGraph, END

# Define the state for the agent
class AgentState(TypedDict):
    company_name: str
    search_urls: List[str]
    scraped_data: str
    final_report: str

def search_discovery(state: AgentState) -> AgentState:
    company_name = state["company_name"]
    tavily_api_key = os.environ.get("TAVILY_API_KEY")

    if not tavily_api_key:
        print("Warning: TAVILY_API_KEY not found. Mocking search results.")
        # For testing without an API key, we return mocked URLs.
        state["search_urls"] = [
            f"https://www.{company_name.lower().replace(' ', '')}.com",
            f"https://www.linkedin.com/company/{company_name.lower().replace(' ', '')}"
        ]
        return state

    url = "https://api.tavily.com/search"
    query = f"{company_name} official website, LinkedIn page, financial business news"
    payload = {
        "api_key": tavily_api_key,
        "query": query,
        "search_depth": "basic",
        "include_answer": False,
        "include_images": False,
        "include_raw_content": False,
        "max_results": 3,
    }
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        results = response.json().get("results", [])
        urls = [res.get("url") for res in results if res.get("url")]
        state["search_urls"] = urls
    except Exception as e:
        print(f"Error during search: {e}")
        state["search_urls"] = []

    return state


def deep_scraper(state: AgentState) -> AgentState:
    urls = state.get("search_urls", [])
    all_text = ""

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    for url in urls:
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')

                # Try to extract meaningful text
                for script in soup(["script", "style"]):
                    script.extract()

                text = soup.get_text(separator=' ', strip=True)

                # Take the first 5000 characters to avoid huge payloads
                all_text += f"\n\n--- Content from {url} ---\n{text[:5000]}"
        except Exception as e:
            print(f"Error scraping {url}: {e}")

    state["scraped_data"] = all_text
    return state


def data_extraction_synthesis(state: AgentState) -> AgentState:
    scraped_data = state.get("scraped_data", "")
    company_name = state.get("company_name", "")
    openai_api_key = os.environ.get("OPENAI_API_KEY")

    if not openai_api_key:
        print("Warning: OPENAI_API_KEY not found. Mocking report generation.")
        state["final_report"] = f"# Corporate Intelligence Report: {company_name}\n\n[Mocked Report due to missing OPENAI_API_KEY]"
        return state

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }

    prompt = f"""
You are an elite Corporate Intelligence Researcher.
Your task is to generate a highly structured, data-driven intelligence report for the company: {company_name}.
You must cross-reference the data, discard marketing fluff, and extract hard facts based strictly on the scraped data below.

Scraped Data:
{scraped_data}

Required Report Structure:

# Executive Summary
A concise, one-paragraph overview of the company.

# Company Profile
- Specialization: What is their exact niche, core technology, or primary service?
- Company Size: Number of employees (provide an exact number or estimated range based on scraped data).
- Headquarters / Key Locations: Primary operational bases.

# Company Performance
- Financials / Growth: Estimated revenue, funding rounds, market share, or notable growth metrics.
- Market Position: Who are their primary competitors?
- Core Products & Offerings: The specific products or services they sell and their target demographic.
- Recent Developments: Key news, leadership changes, or major events from the last 6-12 months.

Constraints:
Ground your entire report strictly in the data you scrape. If a specific metric (like revenue or exact employee count) cannot be found, explicitly state 'Insufficient data found for this metric' rather than guessing.
"""

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that generates intelligence reports based only on provided data."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        report = response.json()["choices"][0]["message"]["content"]
        state["final_report"] = report
    except Exception as e:
        print(f"Error during report generation: {e}")
        state["final_report"] = f"# Error Generating Report\n\nCould not generate report for {company_name}. Error: {e}"

    return state


# Build the LangGraph
def build_graph() -> StateGraph:
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("search_discovery", search_discovery)
    workflow.add_node("deep_scraper", deep_scraper)
    workflow.add_node("data_extraction_synthesis", data_extraction_synthesis)

    # Define edges
    workflow.set_entry_point("search_discovery")
    workflow.add_edge("search_discovery", "deep_scraper")
    workflow.add_edge("deep_scraper", "data_extraction_synthesis")
    workflow.add_edge("data_extraction_synthesis", END)

    # Compile the graph
    app = workflow.compile()
    return app


def main():
    if len(sys.argv) < 2:
        print("Usage: python corporate_intelligence_agent.py \"[Company Name]\"")
        sys.exit(1)

    company_name = sys.argv[1]

    # Initialize the state
    initial_state = {
        "company_name": company_name,
        "search_urls": [],
        "scraped_data": "",
        "final_report": ""
    }

    app = build_graph()

    print(f"Starting investigation for: {company_name}...")

    # Run the graph
    result = app.invoke(initial_state)

    print("\n" + "="*50 + "\n")
    print(result.get("final_report", "No report generated."))
    print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main()
