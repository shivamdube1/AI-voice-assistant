import os
import sys
import json
import requests
from bs4 import BeautifulSoup
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END

class AgentState(TypedDict):
    company_name: str
    search_results: List[Dict[str, str]]
    scraped_data: str
    final_report: str

def search_discovery(state: AgentState) -> AgentState:
    """
    Search & Discovery Node: Use Tavily to find the company's official website,
    LinkedIn page, and recent financial/business news.
    """
    company_name = state["company_name"]
    tavily_api_key = os.environ.get("TAVILY_API_KEY")

    if not tavily_api_key:
        print("Warning: TAVILY_API_KEY not found in environment.")
        return {"search_results": []}

    query = f"{company_name} official website OR LinkedIn OR recent financial business news"

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

    try:
        response = requests.post("https://api.tavily.com/search", json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()

        results = data.get("results", [])
        search_results = [{"url": res.get("url"), "title": res.get("title")} for res in results if res.get("url")]

        return {"search_results": search_results}
    except Exception as e:
        print(f"Error in search_discovery: {e}")
        return {"search_results": []}

def deep_scraper(state: AgentState) -> AgentState:
    """
    Deep Scraper Node: Extract text from the official website and external
    aggregators to specifically hunt for employee headcounts, performance metrics,
    and core specializations.
    """
    search_results = state.get("search_results", [])
    scraped_data_parts = []

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    for result in search_results:
        url = result.get("url")
        if not url:
            continue

        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()

            # Get text
            text = soup.get_text(separator=' ')

            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)

            # Truncate text to avoid blowing up context window (arbitrary limit of 5000 chars per page)
            if len(text) > 5000:
                text = text[:5000] + "... [TRUNCATED]"

            scraped_data_parts.append(f"Source URL: {url}\nContent:\n{text}\n")

        except Exception as e:
            print(f"Error scraping {url}: {e}")

    scraped_data = "\n" + "="*50 + "\n".join(scraped_data_parts)
    return {"scraped_data": scraped_data}

def data_extraction_synthesis(state: AgentState) -> AgentState:
    """
    Data Extraction & Synthesis Node: Pass the raw data through an LLM to extract
    business metrics, filter out noise, and generate a final Markdown report.
    """
    company_name = state["company_name"]
    scraped_data = state.get("scraped_data", "")
    openai_api_key = os.environ.get("OPENAI_API_KEY")

    if not openai_api_key:
        print("Warning: OPENAI_API_KEY not found in environment.")
        return {"final_report": "Error: OPENAI_API_KEY missing."}

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

    user_prompt = f"Target Company: {company_name}\n\nScraped Data:\n{scraped_data}\n\nPlease generate the report based on the scraped data."

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.2
    }

    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()

        report = data["choices"][0]["message"]["content"]
        return {"final_report": report}
    except Exception as e:
        print(f"Error in data_extraction_synthesis: {e}")
        if 'response' in locals() and hasattr(response, 'text'):
            print(f"Response text: {response.text}")
        return {"final_report": "Error generating report."}

# Compile Graph
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
        print("Usage: python corporate_intelligence_agent.py \"[Company Name]\"")
        sys.exit(1)

    company_name = sys.argv[1]

    print(f"Initializing Corporate Intelligence Agent for: {company_name}...\n")

    # Run the graph
    initial_state = {"company_name": company_name}
    final_state = app.invoke(initial_state)

    print("\n" + "="*80)
    print("FINAL REPORT:")
    print("="*80 + "\n")
    print(final_state.get("final_report", "No report generated."))
