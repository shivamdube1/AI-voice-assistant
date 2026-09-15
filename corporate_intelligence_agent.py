import os
import sys
import json
import requests
from bs4 import BeautifulSoup
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END

# Define Agent State
class AgentState(TypedDict):
    company_name: str
    search_urls: List[str]
    scraped_data: str
    final_report: str

# Node 1: Search & Discovery
def search_discovery(state: AgentState) -> Dict:
    company = state["company_name"]
    api_key = os.environ.get("TAVILY_API_KEY")

    if not api_key:
        # Fallback for missing API key (e.g., testing or initial discovery)
        mock_urls = [
            f"https://www.{company.replace(' ', '').lower()}.com",
            f"https://en.wikipedia.org/wiki/{company.replace(' ', '_')}"
        ]
        return {"search_urls": mock_urls}

    query = f"official website {company} OR crunchbase {company} OR financial news {company}"

    headers = {"Content-Type": "application/json"}
    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": "advanced",
        "include_answer": False,
        "include_raw_content": False,
        "max_results": 5
    }

    try:
        response = requests.post("https://api.tavily.com/search", json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        urls = [result["url"] for result in data.get("results", [])]
    except Exception as e:
        print(f"Error during search: {e}")
        urls = []

    return {"search_urls": urls}


# Node 2: Deep Scraper
def deep_scraper(state: AgentState) -> Dict:
    urls = state.get("search_urls", [])
    all_text = ""

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    for url in urls:
        if "linkedin.com" in url.lower():
            # Explicitly skip linkedin to avoid request blocks during standard HTTP operations
            continue

        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, "html.parser")
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.extract()
                text = soup.get_text(separator=' ', strip=True)
                # Quick truncation to avoid context limits
                all_text += f"\n--- Content from {url} ---\n{text[:3000]}\n"
        except Exception as e:
            print(f"Error scraping {url}: {e}")

    return {"scraped_data": all_text}


# Node 3: Data Extraction & Synthesis
def data_extraction_synthesis(state: AgentState) -> Dict:
    company = state["company_name"]
    scraped_data = state.get("scraped_data", "")
    api_key = os.environ.get("OPENAI_API_KEY")

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

    user_prompt = f"Target Company: {company}\n\nScraped Data Context:\n{scraped_data}\n\nPlease generate the report based ONLY on the provided context."

    if not api_key:
        return {"final_report": "Error: OPENAI_API_KEY environment variable not set. Cannot synthesize report."}

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "gpt-4o", # Using a common default since specific model wasn't provided
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.2
    }

    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        report = data["choices"][0]["message"]["content"]
    except Exception as e:
        report = f"Error during synthesis: {e}"

    return {"final_report": report}

# Graph Compilation
builder = StateGraph(AgentState)

builder.add_node("search_discovery", search_discovery)
builder.add_node("deep_scraper", deep_scraper)
builder.add_node("data_extraction_synthesis", data_extraction_synthesis)

builder.set_entry_point("search_discovery")
builder.add_edge("search_discovery", "deep_scraper")
builder.add_edge("deep_scraper", "data_extraction_synthesis")
builder.add_edge("data_extraction_synthesis", END)

app = builder.compile()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python corporate_intelligence_agent.py "[Company Name]"')
        sys.exit(1)

    company_name = sys.argv[1]

    initial_state = {
        "company_name": company_name,
        "search_urls": [],
        "scraped_data": "",
        "final_report": ""
    }

    print(f"Starting intelligence gathering for: {company_name}...\n")

    # Capture final state directly from app.stream()
    final_state = None
    for output in app.stream(initial_state):
        for key, value in output.items():
            print(f"--- Finished node: {key} ---")
            final_state = value # Keep updating to get the last state emitted

    if final_state and "final_report" in final_state:
        print("\n\n" + "="*50)
        print("FINAL REPORT:")
        print("="*50 + "\n")
        print(final_state["final_report"])
    else:
        print("Failed to generate report.")
