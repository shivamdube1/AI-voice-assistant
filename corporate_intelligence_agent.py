import os
import sys
import json
import requests
from bs4 import BeautifulSoup
from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END

class AgentState(TypedDict):
    company_name: str
    urls_to_scrape: List[str]
    scraped_data: str
    report: str

def search_discovery(state: AgentState) -> AgentState:
    company_name = state["company_name"]
    print(f"[{company_name}] Running Search & Discovery...")

    tavily_api_key = os.environ.get("TAVILY_API_KEY")
    if not tavily_api_key:
        return {"urls_to_scrape": []}

    query = f"{company_name} official website, LinkedIn, recent financial business news"

    headers = {
        "Content-Type": "application/json"
    }

    payload = {
        "api_key": tavily_api_key,
        "query": query,
        "search_depth": "advanced",
        "include_answer": False,
        "include_images": False,
        "include_raw_content": False,
        "max_results": 5
    }

    urls = []
    try:
        response = requests.post("https://api.tavily.com/search", headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        urls = [result["url"] for result in data.get("results", [])]
        print(f"Found URLs: {urls}")
    except Exception as e:
        print(f"Error during Tavily search: {e}")

    return {"urls_to_scrape": urls}

def deep_scraper(state: AgentState) -> AgentState:
    urls = state.get("urls_to_scrape", [])
    print(f"Running Deep Scraper on {len(urls)} URLs...")

    scraped_text = ""
    for url in urls:
        print(f"Scraping {url}...")
        try:
            # We add a generic user agent to avoid basic blocks
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')
            # Extract text and clean it up
            text = soup.get_text(separator=' ', strip=True)
            # Limit the amount of text to avoid context window issues
            scraped_text += f"\n\n--- Content from {url} ---\n{text[:5000]}"
        except Exception as e:
            print(f"Error scraping {url}: {e}")

    return {"scraped_data": scraped_text}

def data_extraction_synthesis(state: AgentState) -> AgentState:
    print("Running Data Extraction & Synthesis...")

    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        return {"report": "OPENAI_API_KEY not set. Cannot generate report."}

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

    user_prompt = f"Target Company: {state['company_name']}\n\nScraped Data:\n{state.get('scraped_data', 'No data scraped.')}\n\nGenerate the intelligence report based strictly on the data provided."

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }

    payload = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.2
    }

    report = ""
    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        report = data["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error during OpenAI synthesis: {e}")
        report = f"Error generating report: {e}"

    return {"report": report}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python corporate_intelligence_agent.py \"[Company Name]\"")
        sys.exit(1)

    company = sys.argv[1]

    # Check for API keys
    if not os.environ.get("TAVILY_API_KEY"):
        print("Warning: TAVILY_API_KEY environment variable not set.")
    if not os.environ.get("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY environment variable not set.")

    print(f"Agent started for: {company}")

    # Define the graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("search_discovery", search_discovery)
    workflow.add_node("deep_scraper", deep_scraper)
    workflow.add_node("data_extraction_synthesis", data_extraction_synthesis)

    # Define edges
    workflow.add_edge(START, "search_discovery")
    workflow.add_edge("search_discovery", "deep_scraper")
    workflow.add_edge("deep_scraper", "data_extraction_synthesis")
    workflow.add_edge("data_extraction_synthesis", END)

    # Compile
    app = workflow.compile()

    # Run
    initial_state = {"company_name": company, "urls_to_scrape": [], "scraped_data": "", "report": ""}
    result = app.invoke(initial_state)

    print("\n" + "="*50)
    print("FINAL REPORT:")
    print("="*50 + "\n")
    print(result.get("report", "No report generated."))
