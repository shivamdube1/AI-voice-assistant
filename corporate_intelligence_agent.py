import os
import sys
import json
import requests
from bs4 import BeautifulSoup
from typing import TypedDict, List
from langgraph.graph import StateGraph, END

class AgentState(TypedDict):
    company_name: str
    search_urls: List[str]
    scraped_data: str
    final_report: str

def search_discovery(state: AgentState) -> AgentState:
    """Use Tavily API to find official website, LinkedIn, and news URLs."""
    company_name = state["company_name"]
    tavily_api_key = os.environ.get("TAVILY_API_KEY")

    if not tavily_api_key:
        raise ValueError("TAVILY_API_KEY environment variable is not set.")

    query = f"{company_name} official website OR {company_name} linkedin OR {company_name} recent financial business news"

    try:
        response = requests.post(
            "https://api.tavily.com/search",
            headers={"Content-Type": "application/json"},
            json={
                "api_key": tavily_api_key,
                "query": query,
                "search_depth": "basic",
                "include_answer": False,
                "max_results": 5
            }
        )
        response.raise_for_status()
        data = response.json()

        urls = [result["url"] for result in data.get("results", [])]
        state["search_urls"] = urls
    except Exception as e:
        print(f"Error during search: {e}")
        state["search_urls"] = []

    return state

def deep_scraper(state: AgentState) -> AgentState:
    """Extract text from URLs."""
    urls = state.get("search_urls", [])
    scraped_texts = []

    for url in urls:
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")

            # Extract text, removing script and style elements
            for script_or_style in soup(["script", "style", "nav", "footer", "header"]):
                script_or_style.decompose()

            text = soup.get_text(separator=' ', strip=True)
            scraped_texts.append(f"--- Data from {url} ---\n{text}")
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            continue

    state["scraped_data"] = "\n\n".join(scraped_texts)
    return state

def data_extraction_synthesis(state: AgentState) -> AgentState:
    """Extract metrics using OpenAI API and generate Markdown report."""
    company_name = state["company_name"]
    scraped_data = state.get("scraped_data", "")
    openai_api_key = os.environ.get("OPENAI_API_KEY")

    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")

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

    user_prompt = f"Target Company: {company_name}\n\nScraped Data:\n{scraped_data[:15000]}" # Limit context window just in case

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {openai_api_key}"
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
        data = response.json()

        state["final_report"] = data["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error during synthesis: {e}")
        state["final_report"] = f"Error generating report: {e}"

    return state

# Compile the Graph
workflow = StateGraph(AgentState)

# Define nodes
workflow.add_node("search_discovery", search_discovery)
workflow.add_node("deep_scraper", deep_scraper)
workflow.add_node("data_extraction_synthesis", data_extraction_synthesis)

# Define edges
workflow.set_entry_point("search_discovery")
workflow.add_edge("search_discovery", "deep_scraper")
workflow.add_edge("deep_scraper", "data_extraction_synthesis")
workflow.add_edge("data_extraction_synthesis", END)

# Compile
app = workflow.compile()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python corporate_intelligence_agent.py "[Company Name]"')
        sys.exit(1)

    target_company = sys.argv[1]

    initial_state = AgentState(
        company_name=target_company,
        search_urls=[],
        scraped_data="",
        final_report=""
    )

    print(f"Starting investigation on {target_company}...\n")

    try:
        final_state = app.invoke(initial_state)
        print("\n=== FINAL REPORT ===\n")
        print(final_state["final_report"])
    except Exception as e:
        print(f"Agent failed: {e}")
