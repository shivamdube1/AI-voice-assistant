import os
import sys
import requests
from bs4 import BeautifulSoup
from typing import TypedDict, List
from langgraph.graph import StateGraph, END

# 1. State Definition
class AgentState(TypedDict):
    company_name: str
    search_urls: List[str]
    scraped_data: str
    final_report: str

# 2. Node Functions

def search_discovery(state: AgentState) -> AgentState:
    """
    Search & Discovery Node: Use a search tool (Tavily) to find the company's
    official website, LinkedIn page, and recent financial/business news.
    """
    company_name = state["company_name"]
    tavily_api_key = os.environ.get("TAVILY_API_KEY")

    if not tavily_api_key:
        print("Warning: TAVILY_API_KEY environment variable not set. Returning dummy URL.")
        state["search_urls"] = ["http://example.com"]
        return state

    query = f"{company_name} official website OR LinkedIn OR recent financial business news"

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
        data = response.json()
        urls = [result.get("url") for result in data.get("results", []) if "url" in result]
        state["search_urls"] = urls
    except Exception as e:
        print(f"Error in search_discovery: {e}")
        state["search_urls"] = []

    return state

def deep_scraper(state: AgentState) -> AgentState:
    """
    Deep Scraper Node: Extract text from the URLs specifically hunting for
    employee headcounts, performance metrics, and core specializations.
    """
    urls = state.get("search_urls", [])
    all_text = []

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    for url in urls:
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract basic text, removing scripts and styles
            for script in soup(["script", "style"]):
                script.extract()

            text = soup.get_text(separator=' ', strip=True)
            # Limit the text length to avoid context window issues
            text = text[:5000]
            all_text.append(f"--- Data from {url} ---\n{text}\n")

        except Exception as e:
            print(f"Error scraping {url}: {e}")

    state["scraped_data"] = "\n".join(all_text)
    return state

def data_extraction_synthesis(state: AgentState) -> AgentState:
    """
    Data Extraction & Synthesis Node: Pass the raw data through an LLM to
    extract business metrics, filter noise, and generate the Markdown report.
    """
    company_name = state["company_name"]
    scraped_data = state.get("scraped_data", "")

    openai_api_key = os.environ.get("OPENAI_API_KEY")

    if not openai_api_key:
        print("Warning: OPENAI_API_KEY environment variable not set. Returning dummy report.")
        state["final_report"] = "# Report\nInsufficient data found for this metric."
        return state

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

    user_prompt = f"Target Company: {company_name}\n\nScraped Data:\n{scraped_data}\n\nPlease generate the corporate intelligence report based on the scraped data."

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
        response = requests.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        report = data["choices"][0]["message"]["content"]
        state["final_report"] = report
    except Exception as e:
        print(f"Error in data_extraction_synthesis: {e}")
        state["final_report"] = "Error generating report."

    return state


# 3. Compile Graph Logic
def build_graph() -> StateGraph:
    workflow = StateGraph(AgentState)

    workflow.add_node("search_discovery", search_discovery)
    workflow.add_node("deep_scraper", deep_scraper)
    workflow.add_node("data_extraction_synthesis", data_extraction_synthesis)

    workflow.set_entry_point("search_discovery")
    workflow.add_edge("search_discovery", "deep_scraper")
    workflow.add_edge("deep_scraper", "data_extraction_synthesis")
    workflow.add_edge("data_extraction_synthesis", END)

    return workflow.compile()

def run_agent(company_name: str) -> str:
    app = build_graph()
    initial_state = {"company_name": company_name}
    result = app.invoke(initial_state)
    return result.get("final_report", "No report generated.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python corporate_intelligence_agent.py \"[Company Name]\"")
        sys.exit(1)

    company = sys.argv[1]
    print(f"Running intelligence agent for: {company}...")
    report = run_agent(company)
    print("\n--- FINAL REPORT ---\n")
    print(report)
