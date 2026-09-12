import os
import requests
import json
from bs4 import BeautifulSoup
from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END

# Define the state object for the graph
class AgentState(TypedDict):
    company_name: str
    search_urls: List[str]
    scraped_data: str
    final_report: str

def search_discovery(state: AgentState) -> AgentState:
    """Uses Tavily search to find company URLs (website, LinkedIn, news)."""
    company_name = state["company_name"]
    tavily_api_key = os.environ.get("TAVILY_API_KEY")

    if not tavily_api_key:
        print(f"Warning: TAVILY_API_KEY not found. Using mock URLs for {company_name}.")
        return {
            "company_name": company_name,
            "search_urls": [
                f"https://www.{company_name.lower().replace(' ', '')}.com/about",
                f"https://www.{company_name.lower().replace(' ', '')}.com/careers",
                "https://mocknews.com/recent"
            ]
        }

    query = f"{company_name} official website OR LinkedIn OR recent business news"

    response = requests.post(
        "https://api.tavily.com/search",
        json={
            "api_key": tavily_api_key,
            "query": query,
            "search_depth": "basic",
            "include_answer": False,
            "include_raw_content": False,
            "max_results": 5,
        }
    )

    if response.status_code == 200:
        data = response.json()
        urls = [result["url"] for result in data.get("results", [])]
        return {"company_name": company_name, "search_urls": urls}
    else:
        print(f"Tavily search failed: {response.text}")
        return {"company_name": company_name, "search_urls": []}

def deep_scraper(state: AgentState) -> AgentState:
    """Scrapes raw text from the discovered URLs, skipping LinkedIn."""
    urls = state.get("search_urls", [])
    scraped_text = ""

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    for url in urls:
        if "linkedin.com" in url:
            print(f"Skipping LinkedIn URL to avoid blocks: {url}")
            continue

        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                # Extract text, remove scripts and styles
                for script in soup(["script", "style"]):
                    script.extract()
                text = soup.get_text(separator=' ', strip=True)
                scraped_text += f"\n--- Data from {url} ---\n{text[:2000]}...\n" # Limit text per URL
            else:
                print(f"Failed to scrape {url} - Status Code: {response.status_code}")
        except Exception as e:
            print(f"Error scraping {url}: {e}")

    return {"company_name": state["company_name"], "search_urls": urls, "scraped_data": scraped_text}

def data_extraction_synthesis(state: AgentState) -> AgentState:
    """Uses an LLM to extract metrics and generate a structured Markdown report."""
    company_name = state["company_name"]
    scraped_data = state.get("scraped_data", "")
    openai_api_key = os.environ.get("OPENAI_API_KEY")

    if not openai_api_key:
        print(f"Warning: OPENAI_API_KEY not found. Using mock report for {company_name}.")
        mock_report = f"""# Corporate Intelligence Report: {company_name}

## Executive Summary
This is a mock executive summary for {company_name} due to missing OPENAI_API_KEY.

## Company Profile
- **Specialization**: Insufficient data found for this metric
- **Company Size**: Insufficient data found for this metric
- **Headquarters / Key Locations**: Insufficient data found for this metric

## Company Performance
- **Financials / Growth**: Insufficient data found for this metric
- **Market Position**: Insufficient data found for this metric
- **Core Products & Offerings**: Insufficient data found for this metric
- **Recent Developments**: Insufficient data found for this metric
"""
        return {"company_name": company_name, "final_report": mock_report}

    system_prompt = """You are an elite Corporate Intelligence Researcher. Your mission is to investigate a target company provided by the user, scrape relevant web data, and produce a highly structured, data-driven intelligence report.

Required Report Structure:
# Corporate Intelligence Report: {company_name}

## Executive Summary
A concise, one-paragraph overview of the company.

## Company Profile
- **Specialization**: What is their exact niche, core technology, or primary service?
- **Company Size**: Number of employees (provide an exact number or estimated range based on scraped data).
- **Headquarters / Key Locations**: Primary operational bases.

## Company Performance
- **Financials / Growth**: Estimated revenue, funding rounds, market share, or notable growth metrics.
- **Market Position**: Who are their primary competitors?
- **Core Products & Offerings**: The specific products or services they sell and their target demographic.
- **Recent Developments**: Key news, leadership changes, or major events from the last 6-12 months.

Constraints: Ground your entire report strictly in the data you scrape. If a specific metric (like revenue or exact employee count) cannot be found, explicitly state 'Insufficient data found for this metric' rather than guessing."""

    user_prompt = f"Analyze the following scraped data for {company_name} and generate the intelligence report:\n\n{scraped_data}"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system_prompt.replace("{company_name}", company_name)},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.2
    }

    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        report = response.json()["choices"][0]["message"]["content"]
        return {"company_name": company_name, "final_report": report}
    except Exception as e:
        print(f"Error generating report: {e}")
        return {"company_name": company_name, "final_report": f"# Error\nFailed to generate report: {e}"}

# Build and compile the graph
workflow = StateGraph(AgentState)

workflow.add_node("search_discovery", search_discovery)
workflow.add_node("deep_scraper", deep_scraper)
workflow.add_node("data_extraction_synthesis", data_extraction_synthesis)

workflow.add_edge(START, "search_discovery")
workflow.add_edge("search_discovery", "deep_scraper")
workflow.add_edge("deep_scraper", "data_extraction_synthesis")
workflow.add_edge("data_extraction_synthesis", END)

app = workflow.compile()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        target_company = sys.argv[1]
    else:
        target_company = "OpenAI" # Default for testing

    print(f"Starting Corporate Intelligence Agent for: {target_company}\n")

    initial_state = {"company_name": target_company}

    final_state = None
    for event in app.stream(initial_state):
        for k, v in event.items():
            print(f"--- Completed Node: {k} ---")
            final_state = v

    print("\n" + "="*50 + "\n")
    if final_state:
        print(final_state.get("final_report", "No report generated."))
    else:
        print("No report generated.")
    print("\n" + "="*50 + "\n")
