import os
import sys
import json
import requests
from bs4 import BeautifulSoup
from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END

# Define the state for the graph
class AgentState(TypedDict):
    company_name: str
    search_results: List[str]
    scraped_text: str
    final_report: str

def search_discovery(state: AgentState) -> AgentState:
    """
    Search & Discovery Node: Uses Tavily API to find the company's official website,
    LinkedIn page, and recent financial/business news.
    """
    print("Running Search & Discovery...")
    company_name = state["company_name"]
    tavily_api_key = os.environ.get("TAVILY_API_KEY")

    if not tavily_api_key:
        print("Warning: TAVILY_API_KEY not found. Search may fail.")

    query = f"{company_name} official website, LinkedIn, recent financial business news, employee headcount, revenue"

    url = "https://api.tavily.com/search"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "api_key": tavily_api_key,
        "query": query,
        "include_answer": False,
        "include_raw_content": False,
        "max_results": 5
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        results = response.json().get("results", [])
        urls = [res.get("url") for res in results if res.get("url")]
        return {"search_results": urls}
    except Exception as e:
        print(f"Error during search: {e}")
        return {"search_results": []}

def deep_scraper(state: AgentState) -> AgentState:
    """
    Deep Scraper Node: Extracts text from URLs found in the previous step.
    """
    print("Running Deep Scraper...")
    urls = state.get("search_results", [])
    scraped_texts = []

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    for url in urls:
        try:
            print(f"Scraping {url}...")
            # Set a timeout so we don't hang forever
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()

            text = soup.get_text(separator=' ', strip=True)
            # Take a chunk of text to prevent context limits
            scraped_texts.append(f"Source ({url}):\n" + text[:2000])
        except Exception as e:
            print(f"Failed to scrape {url}: {e}")

    combined_text = "\n\n".join(scraped_texts)
    return {"scraped_text": combined_text}

def data_extraction_synthesis(state: AgentState) -> AgentState:
    """
    Data Extraction & Synthesis Node: Passes the raw data through an LLM to extract targeted business metrics.
    """
    print("Running Data Extraction & Synthesis...")
    scraped_text = state.get("scraped_text", "")
    company_name = state["company_name"]
    openai_api_key = os.environ.get("OPENAI_API_KEY")

    if not openai_api_key:
        print("Warning: OPENAI_API_KEY not found. Synthesis may fail.")

    system_prompt = """You are an elite Corporate Intelligence Researcher. Your mission is to investigate a target company provided by the user, scrape relevant web data, and produce a highly structured, data-driven intelligence report.

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

Constraints: Ground your entire report strictly in the data provided. If a specific metric (like revenue or exact employee count) cannot be found, explicitly state 'Insufficient data found for this metric' rather than guessing. Do not include markdown headers like ```markdown."""

    user_prompt = f"Company to research: {company_name}\n\nHere is the scraped raw data to extract facts from:\n\n{scraped_text}"

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
        return {"final_report": report}
    except Exception as e:
        print(f"Error during synthesis: {e}")
        return {"final_report": f"Error generating report: {e}"}

# Build the Graph
workflow = StateGraph(AgentState)

workflow.add_node("search_discovery", search_discovery)
workflow.add_node("deep_scraper", deep_scraper)
workflow.add_node("data_extraction_synthesis", data_extraction_synthesis)

workflow.add_edge(START, "search_discovery")
workflow.add_edge("search_discovery", "deep_scraper")
workflow.add_edge("deep_scraper", "data_extraction_synthesis")
workflow.add_edge("data_extraction_synthesis", END)

# Compile the graph
app = workflow.compile()

def run_agent(company_name: str):
    print(f"Starting investigation on: {company_name}\n")
    initial_state = {"company_name": company_name}

    # Run the graph
    result = app.invoke(initial_state)

    print("\n" + "="*50)
    print("FINAL INTELLIGENCE REPORT")
    print("="*50 + "\n")
    print(result.get("final_report", "No report generated."))
    print("\n" + "="*50)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python corporate_intelligence_agent.py \"[Company Name]\"")
        sys.exit(1)

    target_company = sys.argv[1]
    run_agent(target_company)
