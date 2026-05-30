import json
import os
import sys
import requests
from typing import TypedDict, Optional
from bs4 import BeautifulSoup
from langgraph.graph import StateGraph, END

class AgentState(TypedDict):
    company_name: str
    search_urls: list[str]
    raw_scraped_data: list[str]
    final_report: Optional[str]


def search_discovery(state: AgentState):
    """
    Search & Discovery Node:
    Uses Tavily API to find the company's official website, LinkedIn page,
    and recent financial/business news.
    """
    company_name = state["company_name"]
    print(f"[*] Running Search & Discovery for: {company_name}")

    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        print("[!] TAVILY_API_KEY is not set.")
        # For testing fallback without API key
        return {"search_urls": []}

    query = f"{company_name} official website, linkedin, recent financial business news"

    response = requests.post(
        "https://api.tavily.com/search",
        json={
            "api_key": tavily_api_key,
            "query": query,
            "search_depth": "basic",
            "max_results": 5
        }
    )

    search_urls = []
    if response.status_code == 200:
        data = response.json()
        search_urls = [result.get("url") for result in data.get("results", []) if result.get("url")]
    else:
        print(f"[!] Tavily search failed with status {response.status_code}")

    return {"search_urls": search_urls}


def deep_scraper(state: AgentState):
    """
    Deep Scraper Node:
    Extract text from the official website and external aggregators to
    hunt for employee headcounts, performance metrics, and core specializations.
    """
    search_urls = state.get("search_urls", [])
    print(f"[*] Running Deep Scraper on {len(search_urls)} URLs...")

    raw_scraped_data = []

    # We will simulate the scraping or do a simple GET request
    for url in search_urls:
        try:
            # We use a user-agent to avoid simple blocks
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers, timeout=5)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                # Extract text, stripping HTML
                text = soup.get_text(separator=' ', strip=True)
                # Keep first 5000 chars to avoid prompt overflow
                raw_scraped_data.append(f"Source: {url}\nContent: {text[:5000]}")
            else:
                raw_scraped_data.append(f"Source: {url}\nContent: [Failed to load, status: {response.status_code}]")
        except Exception as e:
            raw_scraped_data.append(f"Source: {url}\nContent: [Failed to scrape: {str(e)}]")

    return {"raw_scraped_data": raw_scraped_data}


def data_extraction_synthesis(state: AgentState):
    """
    Data Extraction & Synthesis Node:
    Pass raw data through an LLM to extract metrics, filter noise, and generate
    the final Markdown report based on strict guidelines.
    """
    print("[*] Running Data Extraction & Synthesis...")
    company_name = state["company_name"]
    raw_scraped_data = state.get("raw_scraped_data", [])

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        return {"final_report": "[!] OPENAI_API_KEY is not set. Could not generate report."}

    context_text = "\n\n---\n\n".join(raw_scraped_data)
    if not context_text:
        context_text = "No context data could be extracted."

    system_prompt = (
        "You are an elite Corporate Intelligence Researcher. Your mission is to investigate a target company provided by the user, "
        "scrape relevant web data, and produce a highly structured, data-driven intelligence report.\n\n"
        "Required Report Structure:\n"
        "Executive Summary: A concise, one-paragraph overview of the company.\n\n"
        "Company Profile:\n"
        "- Specialization: What is their exact niche, core technology, or primary service?\n"
        "- Company Size: Number of employees (provide an exact number or estimated range based on scraped data).\n"
        "- Headquarters / Key Locations: Primary operational bases.\n\n"
        "Company Performance:\n"
        "- Financials / Growth: Estimated revenue, funding rounds, market share, or notable growth metrics.\n"
        "- Market Position: Who are their primary competitors?\n"
        "- Core Products & Offerings: The specific products or services they sell and their target demographic.\n"
        "- Recent Developments: Key news, leadership changes, or major events from the last 6-12 months.\n\n"
        "Constraints: Ground your entire report strictly in the data you scrape. If a specific metric (like revenue or exact employee count) "
        "cannot be found, explicitly state 'Insufficient data found for this metric' rather than guessing."
    )

    user_prompt = f"Target Company: {company_name}\n\nScraped Data:\n{context_text}\n\nGenerate the structured intelligence report."

    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {openai_api_key}",
            "Content-Type": "application/json"
        },
        json={
            "model": "gpt-4o",  # or gpt-3.5-turbo
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.2
        }
    )

    final_report = "Error generating report."
    if response.status_code == 200:
        data = response.json()
        final_report = data["choices"][0]["message"]["content"]
    else:
        print(f"[!] OpenAI API request failed with status {response.status_code}: {response.text}")

    return {"final_report": final_report}


def compile_graph():
    """
    Compiles the LangGraph workflow.
    """
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("search_discovery", search_discovery)
    workflow.add_node("deep_scraper", deep_scraper)
    workflow.add_node("data_extraction_synthesis", data_extraction_synthesis)

    # Set entry point
    workflow.set_entry_point("search_discovery")

    # Add edges
    workflow.add_edge("search_discovery", "deep_scraper")
    workflow.add_edge("deep_scraper", "data_extraction_synthesis")
    workflow.add_edge("data_extraction_synthesis", END)

    # Compile graph
    return workflow.compile()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python corporate_intelligence_agent.py "[Company Name]"')
        sys.exit(1)

    target_company = sys.argv[1]

    print(f"\n==============================================")
    print(f"Starting Corporate Intelligence Agent for: {target_company}")
    print(f"==============================================\n")

    # Initialize the graph
    graph = compile_graph()

    # Initial state
    initial_state = {
        "company_name": target_company,
        "search_urls": [],
        "raw_scraped_data": [],
        "final_report": None
    }

    # Run the graph
    result = graph.invoke(initial_state)

    print(f"\n==============================================")
    print(f"FINAL REPORT")
    print(f"==============================================\n")
    if result and "final_report" in result:
        print(result["final_report"])
    else:
        print("Report generation failed or no report found in result.")
