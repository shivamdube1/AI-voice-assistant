import os
import sys
import json
import requests
from typing import TypedDict, List
from bs4 import BeautifulSoup
from langgraph.graph import StateGraph, START, END

# Define the State
class AgentState(TypedDict):
    company_name: str
    search_results: List[str]
    scraped_texts: List[str]
    report: str

def search_discovery(state: AgentState) -> AgentState:
    """
    Search & Discovery Node: Use Tavily API to find the company's official website,
    LinkedIn page, and recent financial/business news.
    """
    company_name = state["company_name"]
    tavily_api_key = os.environ.get("TAVILY_API_KEY")

    if not tavily_api_key:
        print("Warning: TAVILY_API_KEY not found in environment variables.")
        return {"search_results": []}

    query = f"{company_name} official website OR LinkedIn OR financial news OR business news"

    headers = {"Content-Type": "application/json"}
    payload = {
        "api_key": tavily_api_key,
        "query": query,
        "search_depth": "advanced",
        "include_answer": False,
        "include_images": False,
        "include_raw_content": False,
        "max_results": 5
    }

    try:
        response = requests.post("https://api.tavily.com/search", headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

        urls = [result["url"] for result in data.get("results", [])]
        return {"search_results": urls}
    except Exception as e:
        print(f"Error during search discovery: {e}")
        return {"search_results": []}

def deep_scraper(state: AgentState) -> AgentState:
    """
    Deep Scraper Node: Extract text from the official website and external aggregators
    to specifically hunt for employee headcounts, performance metrics, and core specializations.
    """
    urls = state.get("search_results", [])
    scraped_texts = []

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    for url in urls:
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')
            # Extract text and remove extra whitespace
            text = soup.get_text(separator=' ', strip=True)

            # Limit the text length to avoid token limits for LLM
            scraped_texts.append(text[:5000])
        except Exception as e:
            print(f"Error scraping {url}: {e}")

    return {"scraped_texts": scraped_texts}

def data_extraction_synthesis(state: AgentState) -> AgentState:
    """
    Data Extraction & Synthesis Node: Pass the raw data through an LLM to extract
    the targeted business metrics, filter out noise, and generate a final Markdown report.
    """
    company_name = state["company_name"]
    scraped_texts = state.get("scraped_texts", [])
    openai_api_key = os.environ.get("OPENAI_API_KEY")

    if not openai_api_key:
        print("Warning: OPENAI_API_KEY not found in environment variables.")
        return {"report": "Error: OPENAI_API_KEY not provided."}

    combined_text = "\n\n---\n\n".join(scraped_texts)

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

    user_prompt = f"Target Company: {company_name}\n\nScraped Data:\n{combined_text}\n\nGenerate the intelligence report based on the provided data and required structure."

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

    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

        report = data["choices"][0]["message"]["content"]
        return {"report": report}
    except Exception as e:
        print(f"Error during data extraction and synthesis: {e}")
        return {"report": f"Error during synthesis: {str(e)}"}

def build_graph():
    """Build and compile the LangGraph StateGraph."""
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("search_discovery", search_discovery)
    workflow.add_node("deep_scraper", deep_scraper)
    workflow.add_node("data_extraction_synthesis", data_extraction_synthesis)

    # Add edges
    workflow.add_edge(START, "search_discovery")
    workflow.add_edge("search_discovery", "deep_scraper")
    workflow.add_edge("deep_scraper", "data_extraction_synthesis")
    workflow.add_edge("data_extraction_synthesis", END)

    # Compile the graph
    app = workflow.compile()
    return app

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python corporate_intelligence_agent.py "[Company Name]"')
        sys.exit(1)

    company_name = sys.argv[1]

    app = build_graph()

    initial_state = {
        "company_name": company_name,
        "search_results": [],
        "scraped_texts": [],
        "report": ""
    }

    print(f"Running Corporate Intelligence Agent for: {company_name}...")

    final_state = None
    for s in app.stream(initial_state):
        if "search_discovery" in s:
            print("Found URLs:", s["search_discovery"].get("search_results", []))
        elif "deep_scraper" in s:
            print(f"Scraped content from {len(s['deep_scraper'].get('scraped_texts', []))} pages.")
        elif "data_extraction_synthesis" in s:
            print("Report generated successfully.")
            final_state = s["data_extraction_synthesis"]

    print("\n--- Corporate Intelligence Report ---\n")
    if final_state:
        print(final_state.get("report", "No report generated."))
    else:
        print("No report generated.")
