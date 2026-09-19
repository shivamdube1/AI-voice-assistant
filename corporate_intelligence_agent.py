import os
import json
import argparse
import requests
from typing import TypedDict, List
from bs4 import BeautifulSoup
from langgraph.graph import StateGraph, END

# Define the state for the agent
class AgentState(TypedDict):
    company_name: str
    search_urls: List[str]
    scraped_data: str
    final_report: str

def search_discovery(state: AgentState) -> AgentState:
    """
    Use a search tool (e.g., Tavily) to find the company's official website,
    LinkedIn page, and recent financial/business news.
    """
    company_name = state.get("company_name", "")
    tavily_api_key = os.environ.get("TAVILY_API_KEY")

    urls = []

    if tavily_api_key:
        query = f"official website linkedin page financial business news for {company_name}"
        try:
            response = requests.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": tavily_api_key,
                    "query": query,
                    "search_depth": "advanced",
                    "include_raw_content": False
                },
                timeout=10
            )
            response.raise_for_status()
            results = response.json().get("results", [])
            urls = [result.get("url") for result in results if result.get("url")]
        except Exception as e:
            print(f"Error during Tavily search: {e}")
    else:
        # Fallback to mock URLs if API key is not found
        urls = [
            f"https://www.{company_name.lower().replace(' ', '')}.com",
            f"https://www.linkedin.com/company/{company_name.lower().replace(' ', '')}",
            f"https://news.ycombinator.com/item?id={company_name.lower().replace(' ', '')}"
        ]

    return {"search_urls": urls}

def deep_scraper(state: AgentState) -> AgentState:
    """
    Extract text from the official website (About Us, Careers) and external aggregators
    to specifically hunt for employee headcounts, performance metrics (revenue, growth),
    and core specializations.
    """
    urls = state.get("search_urls", [])
    scraped_text = ""

    for url in urls:
        if "linkedin.com" in url:
            # Skip linkedin URLs to avoid request blocks
            continue

        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract basic text from paragraphs and headings
            text_blocks = [p.get_text() for p in soup.find_all(['p', 'h1', 'h2', 'h3'])]
            scraped_text += f"\n--- Content from {url} ---\n"
            scraped_text += "\n".join(text_blocks[:50]) # Limit to first 50 blocks to avoid huge text

        except Exception as e:
            print(f"Error scraping {url}: {e}")

    return {"scraped_data": scraped_text}

def data_extraction_synthesis(state: AgentState) -> AgentState:
    """
    Pass the raw data through an LLM to extract the targeted business metrics,
    filter out noise, and generate a final Markdown report.
    """
    scraped_data = state.get("scraped_data", "")
    company_name = state.get("company_name", "")
    openai_api_key = os.environ.get("OPENAI_API_KEY")

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

Constraints: Ground your entire report strictly in the data you scrape. If a specific metric (like revenue or exact employee count) cannot be found, explicitly state 'Insufficient data found for this metric' rather than guessing."""

    user_prompt = f"Target Company: {company_name}\n\nScraped Data:\n{scraped_data}"

    if openai_api_key:
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {openai_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-4-turbo-preview",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.2
                },
                timeout=30
            )
            response.raise_for_status()
            report = response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            report = f"Error during LLM synthesis: {e}\n\nRaw Data:\n{scraped_data}"
    else:
        # Fallback if no OpenAI key is available
        report = f"# Intelligence Report: {company_name}\n\n**Note**: OpenAI API key missing. Returning raw data summary.\n\n## Executive Summary\nInsufficient data found for this metric\n\n## Company Profile\n- Specialization: Insufficient data found for this metric\n- Company Size: Insufficient data found for this metric\n- Headquarters / Key Locations: Insufficient data found for this metric\n\n## Company Performance\n- Financials / Growth: Insufficient data found for this metric\n- Market Position: Insufficient data found for this metric\n- Core Products & Offerings: Insufficient data found for this metric\n- Recent Developments: Insufficient data found for this metric"

    return {"final_report": report}

def create_agent():
    workflow = StateGraph(AgentState)

    workflow.add_node("search_discovery", search_discovery)
    workflow.add_node("deep_scraper", deep_scraper)
    workflow.add_node("data_extraction_synthesis", data_extraction_synthesis)

    workflow.set_entry_point("search_discovery")
    workflow.add_edge("search_discovery", "deep_scraper")
    workflow.add_edge("deep_scraper", "data_extraction_synthesis")
    workflow.add_edge("data_extraction_synthesis", END)

    app = workflow.compile()
    return app

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Corporate Intelligence Agent")
    parser.add_argument("company_name", type=str, help="The name of the target company")
    args = parser.parse_args()

    app = create_agent()

    # Avoid double execution: directly capture from the stream loop
    initial_state = {"company_name": args.company_name}
    final_state = None
    for output in app.stream(initial_state):
        for node_name, state in output.items():
            final_state = state

    if final_state and "final_report" in final_state:
        print("\n" + "="*50 + "\n")
        print(final_state["final_report"])
        print("\n" + "="*50 + "\n")
    else:
        print("Failed to generate report.")
