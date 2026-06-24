import os
import sys
import json
import requests
from bs4 import BeautifulSoup
from typing import TypedDict, List
from langgraph.graph import StateGraph, END

# Define state dictionary
class AgentState(TypedDict):
    company_name: str
    search_results: List[str]  # URLs to scrape
    scraped_data: str          # Raw text from URLs
    report: str                # Final markdown report

# Nodes
def search_discovery(state: AgentState):
    """
    Search & Discovery Node: Use Tavily to find the company's official website,
    LinkedIn page, and recent financial/business news.
    """
    company_name = state["company_name"]
    tavily_api_key = os.environ.get("TAVILY_API_KEY")
    if not tavily_api_key:
        print("Warning: TAVILY_API_KEY not set.")
        return {"search_results": []}

    query = f"{company_name} official website, LinkedIn page, and recent financial/business news"

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
        response = requests.post("https://api.tavily.com/search", json=payload, headers=headers)
        response.raise_for_status()
        results = response.json().get("results", [])
        urls = [result["url"] for result in results if "url" in result]
        return {"search_results": urls}
    except Exception as e:
        print(f"Error during search: {e}")
        return {"search_results": []}

def deep_scraper(state: AgentState):
    """
    Deep Scraper Node: Extract text from URLs to specifically hunt for
    employee headcounts, performance metrics (revenue, growth), and core specializations.
    """
    urls = state["search_results"]
    scraped_data = []

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    for url in urls:
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                # Extract text, removing scripts and styles
                for script in soup(["script", "style"]):
                    script.extract()
                text = soup.get_text(separator=' ', strip=True)
                # Keep it somewhat constrained to avoid massive context
                scraped_data.append(f"Source: {url}\nContent: {text[:5000]}")
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            continue

    return {"scraped_data": "\n\n".join(scraped_data)}

def data_extraction_synthesis(state: AgentState):
    """
    Data Extraction & Synthesis Node: Pass raw data to an LLM to generate the final Markdown report.
    """
    company_name = state["company_name"]
    scraped_data = state.get("scraped_data", "")

    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        print("Warning: OPENAI_API_KEY not set.")
        return {"report": "OPENAI_API_KEY missing. Cannot generate report."}

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

    user_prompt = f"Target Company: {company_name}\n\nScraped Data:\n{scraped_data}\n\nGenerate the report based on the provided constraints and structure."

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }

    payload = {
        "model": "gpt-4o",  # or gpt-3.5-turbo, gpt-4
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
        print(f"Error during synthesis: {e}")
        return {"report": f"Error generating report: {e}"}

# Build the Graph
def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("search_discovery", search_discovery)
    workflow.add_node("deep_scraper", deep_scraper)
    workflow.add_node("data_extraction_synthesis", data_extraction_synthesis)

    workflow.set_entry_point("search_discovery")
    workflow.add_edge("search_discovery", "deep_scraper")
    workflow.add_edge("deep_scraper", "data_extraction_synthesis")
    workflow.add_edge("data_extraction_synthesis", END)

    return workflow.compile()

def run_agent(company_name: str):
    graph = build_graph()
    initial_state = {"company_name": company_name, "search_results": [], "scraped_data": "", "report": ""}

    print(f"Running Corporate Intelligence Agent for: {company_name}...\n")

    # Run the graph
    final_state = graph.invoke(initial_state)

    print("--- Final Report ---")
    print(final_state.get("report", "No report generated."))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python corporate_intelligence_agent.py "[Company Name]"')
        sys.exit(1)

    target_company = sys.argv[1]
    run_agent(target_company)
