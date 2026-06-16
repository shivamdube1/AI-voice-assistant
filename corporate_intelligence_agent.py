import os
import sys
import json
import requests
from typing import TypedDict, List, Dict, Any
from bs4 import BeautifulSoup
from langgraph.graph import StateGraph, END

# Define the state for the LangGraph
class AgentState(TypedDict):
    company_name: str
    search_results: List[Dict[str, Any]]
    scraped_data: str
    final_report: str

def search_discovery(state: AgentState) -> AgentState:
    """Node to find the company's official website, LinkedIn page, and recent financial/business news."""
    print(f"[*] Running Search & Discovery for {state['company_name']}...")
    tavily_api_key = os.environ.get("TAVILY_API_KEY")
    if not tavily_api_key:
        print("Warning: TAVILY_API_KEY not found. Returning empty search results.")
        state["search_results"] = []
        return state

    query = f"{state['company_name']} official website OR LinkedIn OR recent financial business news"

    url = "https://api.tavily.com/search"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "api_key": tavily_api_key,
        "query": query,
        "search_depth": "advanced",
        "include_raw_content": True,
        "max_results": 5
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        results = response.json().get("results", [])
        state["search_results"] = results
    except Exception as e:
        print(f"Error during search: {e}")
        state["search_results"] = []

    return state

def deep_scraper(state: AgentState) -> AgentState:
    """Node to extract text from websites found in search, focusing on data points."""
    print(f"[*] Running Deep Scraper for {state['company_name']}...")
    all_extracted_text = []

    for result in state.get("search_results", []):
        url = result.get("url")
        # Prefer raw content from Tavily if available
        if result.get("raw_content"):
            content = result["raw_content"]
            # Clean up the raw content a bit with BeautifulSoup
            soup = BeautifulSoup(content, "html.parser")
            text = soup.get_text(separator=' ', strip=True)
            all_extracted_text.append(f"Source: {url}\n{text}")
        elif url:
            try:
                # Fallback to direct request
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, "html.parser")
                    text = soup.get_text(separator=' ', strip=True)
                    all_extracted_text.append(f"Source: {url}\n{text}")
            except Exception as e:
                print(f"Failed to scrape {url}: {e}")

    state["scraped_data"] = "\n\n---\n\n".join(all_extracted_text)
    # If no data was scraped but there were search results, at least pass the snippet
    if not state["scraped_data"] and state.get("search_results"):
        snippets = [f"Source: {r.get('url')}\n{r.get('content')}" for r in state["search_results"]]
        state["scraped_data"] = "\n\n---\n\n".join(snippets)

    return state

def data_extraction_synthesis(state: AgentState) -> AgentState:
    """Node to pass raw data through an LLM to generate the final report."""
    print(f"[*] Running Data Extraction & Synthesis for {state['company_name']}...")
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        print("Warning: OPENAI_API_KEY not found. Returning error message.")
        state["final_report"] = "Error: OPENAI_API_KEY not set. Cannot generate report."
        return state

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

Constraints: Ground your entire report strictly in the data you scrape. If a specific metric (like revenue or exact employee count) cannot be found, explicitly state 'Insufficient data found for this metric' rather than guessing. Output the final report in Markdown format."""

    user_prompt = f"Target Company: {state['company_name']}\n\nHere is the scraped raw data to extract from:\n\n{state['scraped_data']}"

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }
    data = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.2
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        report = response.json()["choices"][0]["message"]["content"]
        state["final_report"] = report
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        state["final_report"] = f"Error generating report: {e}"

    return state

def main():
    if len(sys.argv) < 2:
        print('Usage: python corporate_intelligence_agent.py "[Company Name]"')
        sys.exit(1)

    company_name = sys.argv[1]

    # Build the graph
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

    # Initialize state
    initial_state = {
        "company_name": company_name,
        "search_results": [],
        "scraped_data": "",
        "final_report": ""
    }

    print(f"Starting Corporate Intelligence Agent for: {company_name}")
    print("-" * 50)

    # Run the graph
    try:
        final_state = app.invoke(initial_state)
        print("\n" + "=" * 50)
        print("FINAL REPORT")
        print("=" * 50 + "\n")
        print(final_state["final_report"])
    except Exception as e:
        print(f"An error occurred during execution: {e}")

if __name__ == "__main__":
    main()
