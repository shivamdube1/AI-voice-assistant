import os
import requests
import sys
from bs4 import BeautifulSoup
from typing import TypedDict, List
from langgraph.graph import StateGraph, END

class AgentState(TypedDict, total=False):
    company_name: str
    search_results: List[dict]
    scraped_data: str
    report: str

def search_discovery(state: AgentState) -> AgentState:
    company_name = state.get("company_name", "")
    tavily_api_key = os.environ.get("TAVILY_API_KEY")

    # Query for official website, LinkedIn, and recent financial/business news
    query = f"{company_name} official website OR LinkedIn page OR recent financial business news"

    payload = {
        "api_key": tavily_api_key,
        "query": query,
        "search_depth": "advanced",
        "max_results": 5
    }

    response = requests.post("https://api.tavily.com/search", json=payload)
    if response.status_code == 200:
        results = response.json().get("results", [])
    else:
        results = []

    return {"search_results": results}

def deep_scraper(state: AgentState) -> AgentState:
    search_results = state.get("search_results", [])
    scraped_texts = []

    for result in search_results:
        url = result.get("url")
        if url:
            try:
                # Add headers to simulate a real browser to avoid being blocked
                headers = {'User-Agent': 'Mozilla/5.0'}
                response = requests.get(url, headers=headers, timeout=10)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, "html.parser")
                    # Extract text, removing script and style elements
                    for script in soup(["script", "style"]):
                        script.decompose()
                    text = soup.get_text(separator=' ', strip=True)
                    # Limit the amount of text to avoid context limits
                    scraped_texts.append(f"Source: {url}\n{text[:5000]}")
            except Exception as e:
                # If scraping fails for a URL, simply ignore and proceed
                pass

    combined_data = "\n\n".join(scraped_texts)
    return {"scraped_data": combined_data}

def data_extraction_synthesis(state: AgentState) -> AgentState:
    company_name = state.get("company_name", "")
    scraped_data = state.get("scraped_data", "")
    openai_api_key = os.environ.get("OPENAI_API_KEY")

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

    user_prompt = f"Target Company: {company_name}\n\nScraped Data:\n{scraped_data}\n\nPlease generate the comprehensive, structured research report based on the provided scraped data."

    headers = {
        "Authorization": f"Bearer {openai_api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    if response.status_code == 200:
        report = response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
    else:
        report = "Failed to generate report."

    return {"report": report}

# Build and compile the LangGraph workflow
workflow = StateGraph(AgentState)

workflow.add_node("search_discovery", search_discovery)
workflow.add_node("deep_scraper", deep_scraper)
workflow.add_node("data_extraction_synthesis", data_extraction_synthesis)

workflow.set_entry_point("search_discovery")
workflow.add_edge("search_discovery", "deep_scraper")
workflow.add_edge("deep_scraper", "data_extraction_synthesis")
workflow.add_edge("data_extraction_synthesis", END)

app = workflow.compile()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python corporate_intelligence_agent.py "[Company Name]"')
        sys.exit(1)

    target_company = sys.argv[1]

    print(f"Running Corporate Intelligence Agent for: {target_company}...\n")

    inputs = {"company_name": target_company}

    final_state = {}
    for output in app.stream(inputs):
        for node_name, state in output.items():
            print(f"--- Finished node: {node_name} ---")
            final_state = state

    print("\n" + "="*50 + "\n")
    print(final_state.get("report", ""))
