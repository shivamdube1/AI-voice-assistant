import os
import sys
from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END

class AgentState(TypedDict):
    company_name: str
    search_queries: List[str]
    urls_to_scrape: List[str]
    scraped_data: str
    final_report: str

import requests

def search_discovery(state: AgentState) -> AgentState:
    print(f"--> [Search & Discovery] Finding sources for {state['company_name']}...")
    company_name = state["company_name"]
    tavily_api_key = os.environ.get("TAVILY_API_KEY")

    # We will search for official site, linkedin, and recent news
    queries = [
        f"{company_name} official website about us",
        f"{company_name} linkedin company profile",
        f"{company_name} recent financial business news"
    ]

    urls_to_scrape = []

    for query in queries:
        try:
            response = requests.post(
                "https://api.tavily.com/search",
                headers={"Content-Type": "application/json"},
                json={
                    "api_key": tavily_api_key,
                    "query": query,
                    "search_depth": "basic",
                    "max_results": 2
                }
            )
            response.raise_for_status()
            data = response.json()
            for result in data.get("results", []):
                if result.get("url") and result["url"] not in urls_to_scrape:
                    urls_to_scrape.append(result["url"])
        except Exception as e:
            print(f"Error searching for {query}: {e}")

    return {"search_queries": queries, "urls_to_scrape": urls_to_scrape}

from bs4 import BeautifulSoup

def deep_scraper(state: AgentState) -> AgentState:
    print(f"--> [Deep Scraper] Scraping content from {len(state['urls_to_scrape'])} URLs...")
    scraped_texts = []

    # We will set a generic user agent to avoid being blocked by simple checks
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    for url in state["urls_to_scrape"]:
        try:
            # Add a small timeout to not hang too long on bad sites
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                # Remove scripts and styles
                for script in soup(["script", "style"]):
                    script.extract()

                text = soup.get_text(separator=' ')
                # Simple cleanup: collapse whitespace
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = ' '.join(chunk for chunk in chunks if chunk)

                # Limit size per page to avoid context window explosion
                scraped_texts.append(f"Source: {url}\nContent: {text[:5000]}")
            else:
                print(f"Failed to fetch {url}: Status {response.status_code}")
        except Exception as e:
            print(f"Error scraping {url}: {e}")

    combined_data = "\n\n---\n\n".join(scraped_texts)
    return {"scraped_data": combined_data}

import json

def data_extraction_synthesis(state: AgentState) -> AgentState:
    print(f"--> [Data Extraction & Synthesis] Generating report for {state['company_name']}...")
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

    user_prompt = f"Target Company: {state['company_name']}\n\nScraped Data:\n{state['scraped_data']}\n\nGenerate the structured intelligence report based ONLY on the provided scraped data."

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {openai_api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "gpt-4o",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.2
            }
        )
        response.raise_for_status()
        result = response.json()
        final_report = result["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error generating report: {e}")
        final_report = f"Failed to generate report: {e}"

    return {"final_report": final_report}

def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("search_discovery", search_discovery)
    workflow.add_node("deep_scraper", deep_scraper)
    workflow.add_node("data_extraction_synthesis", data_extraction_synthesis)

    workflow.add_edge(START, "search_discovery")
    workflow.add_edge("search_discovery", "deep_scraper")
    workflow.add_edge("deep_scraper", "data_extraction_synthesis")
    workflow.add_edge("data_extraction_synthesis", END)

    return workflow.compile()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python corporate_intelligence_agent.py "[Company Name]"')
        sys.exit(1)

    company_name = sys.argv[1]

    # Check for required API keys
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is missing.")
        sys.exit(1)
    if not os.environ.get("TAVILY_API_KEY"):
        print("Error: TAVILY_API_KEY environment variable is missing.")
        sys.exit(1)

    app = build_graph()
    initial_state = {
        "company_name": company_name,
        "search_queries": [],
        "urls_to_scrape": [],
        "scraped_data": "",
        "final_report": ""
    }

    # Run the graph
    print(f"Starting research on {company_name}...")
    final_state = app.invoke(initial_state)

    # Print the final report
    print("\n\n" + "="*80 + "\n")
    print(final_state.get("final_report", "Report generation failed."))
    print("\n" + "="*80 + "\n")
