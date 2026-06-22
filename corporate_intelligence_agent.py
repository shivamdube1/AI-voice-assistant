import os
import sys
import json
import requests
from bs4 import BeautifulSoup
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END


class AgentState(TypedDict):
    company_name: str
    search_results: List[Dict[str, Any]]
    scraped_data: str
    final_report: str


def search_discovery(state: AgentState) -> AgentState:
    """
    Search & Discovery Node:
    Use a search tool (Tavily) to find the company's official website, LinkedIn page,
    and recent financial/business news.
    """
    company_name = state["company_name"]
    tavily_api_key = os.environ.get("TAVILY_API_KEY")

    if not tavily_api_key:
        print("Warning: TAVILY_API_KEY environment variable not set. Returning empty search results.")
        return {**state, "search_results": []}

    queries = [
        f"{company_name} official website",
        f"{company_name} LinkedIn profile",
        f"{company_name} recent financial business news"
    ]

    all_results = []

    for query in queries:
        try:
            response = requests.post(
                "https://api.tavily.com/search",
                json={"query": query, "include_raw_content": False, "max_results": 2},
                headers={"Authorization": f"Bearer {tavily_api_key}"}
            )
            response.raise_for_status()
            data = response.json()
            if "results" in data:
                all_results.extend(data["results"])
        except Exception as e:
            print(f"Error searching for '{query}': {e}")

    return {**state, "search_results": all_results}


def deep_scraper(state: AgentState) -> AgentState:
    """
    Deep Scraper Node:
    Extract text from the official website (About Us, Careers) and external aggregators
    to specifically hunt for employee headcounts, performance metrics (revenue, growth),
    and core specializations.
    """
    search_results = state.get("search_results", [])
    scraped_texts = []

    # Extract URLs
    urls = [result["url"] for result in search_results if "url" in result]

    # We will limit to 5 URLs to prevent excessive scraping and token usage
    urls = urls[:5]

    for url in urls:
        try:
            # Adding timeout and user-agent to avoid getting blocked easily
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()

            text = soup.get_text(separator=' ', strip=True)

            # Chunking text if it's too long (taking first ~5000 characters to save context window)
            if len(text) > 5000:
                text = text[:5000] + "..."

            scraped_texts.append(f"Source: {url}\nContent: {text}")
        except Exception as e:
            print(f"Error scraping {url}: {e}")

    combined_scraped_data = "\n\n".join(scraped_texts)
    return {**state, "scraped_data": combined_scraped_data}


def data_extraction_synthesis(state: AgentState) -> AgentState:
    """
    Data Extraction & Synthesis Node:
    Pass the raw data through an LLM to extract the targeted business metrics,
    filter out noise, and generate a final Markdown report.
    """
    company_name = state["company_name"]
    scraped_data = state.get("scraped_data", "")
    openai_api_key = os.environ.get("OPENAI_API_KEY")

    if not openai_api_key:
        print("Warning: OPENAI_API_KEY environment variable not set. Returning empty report.")
        return {**state, "final_report": "Error: OPENAI_API_KEY not set."}

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

    user_prompt = f"Target Company: {company_name}\n\nScraped Data:\n{scraped_data}\n\nPlease generate the required structured Markdown report based ONLY on the provided scraped data."

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            json={
                "model": "gpt-4o-mini", # Using gpt-4o-mini for speed and cost-effectiveness while retaining capability
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            },
            headers={
                "Authorization": f"Bearer {openai_api_key}",
                "Content-Type": "application/json"
            }
        )
        response.raise_for_status()
        data = response.json()
        final_report = data["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        final_report = f"Error generating report: {e}"

    return {**state, "final_report": final_report}


def build_graph():
    """
    Builds and returns the compiled LangGraph workflow.
    """
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("search_discovery", search_discovery)
    workflow.add_node("deep_scraper", deep_scraper)
    workflow.add_node("data_extraction_synthesis", data_extraction_synthesis)

    # Define edges
    workflow.set_entry_point("search_discovery")
    workflow.add_edge("search_discovery", "deep_scraper")
    workflow.add_edge("deep_scraper", "data_extraction_synthesis")
    workflow.add_edge("data_extraction_synthesis", END)

    # Compile
    return workflow.compile()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python corporate_intelligence_agent.py \"[Company Name]\"")
        sys.exit(1)

    target_company = sys.argv[1]

    # Check for API keys
    if not os.environ.get("TAVILY_API_KEY") or not os.environ.get("OPENAI_API_KEY"):
        print("WARNING: TAVILY_API_KEY and OPENAI_API_KEY environment variables are required for live execution.")

    app = build_graph()

    initial_state = {
        "company_name": target_company,
        "search_results": [],
        "scraped_data": "",
        "final_report": ""
    }

    print(f"Starting analysis for: {target_company}...\n")

    try:
        result = app.invoke(initial_state)
        print("\n--- Final Report ---\n")
        print(result["final_report"])
    except Exception as e:
        print(f"An error occurred during execution: {e}")
