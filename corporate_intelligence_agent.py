import os
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
    company_name = state["company_name"]
    tavily_api_key = os.environ.get("TAVILY_API_KEY")

    if not tavily_api_key:
        # Fallback to mock URLs if key is not present
        state["search_urls"] = [
            f"https://mockurl.com/{company_name.lower().replace(' ', '')}",
            f"https://mockurl.com/news/{company_name.lower().replace(' ', '')}"
        ]
        return state

    query = f"{company_name} official website linkedin recent financial business news"

    headers = {
        "Content-Type": "application/json"
    }

    data = {
        "api_key": tavily_api_key,
        "query": query,
        "search_depth": "basic",
        "include_answer": False,
        "include_images": False,
        "include_raw_content": False,
        "max_results": 5
    }

    try:
        response = requests.post("https://api.tavily.com/search", headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        urls = [item.get("url") for item in result.get("results", []) if item.get("url")]
        state["search_urls"] = urls
    except Exception as e:
        print(f"Tavily search failed: {e}")
        state["search_urls"] = []

    return state

def deep_scraper(state: AgentState) -> AgentState:
    urls = state.get("search_urls", [])
    scraped_texts = []

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    for url in urls:
        if "linkedin.com" in url:
            continue

        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")

            # Remove script and style elements
            for script_or_style in soup(["script", "style"]):
                script_or_style.decompose()

            text = soup.get_text(separator=' ', strip=True)
            if text:
                scraped_texts.append(f"Source: {url}\n{text[:5000]}") # Keep up to 5000 chars per source to avoid context limits

        except Exception as e:
            print(f"Failed to scrape {url}: {e}")

    state["scraped_data"] = "\n\n---\n\n".join(scraped_texts)
    return state

def data_extraction_synthesis(state: AgentState) -> AgentState:
    company_name = state["company_name"]
    scraped_data = state.get("scraped_data", "")
    openai_api_key = os.environ.get("OPENAI_API_KEY")

    if not openai_api_key:
        state["final_report"] = f"# Corporate Intelligence Report: {company_name}\n\nError: OPENAI_API_KEY not found."
        return state

    system_prompt = """You are an elite Corporate Intelligence Researcher. Your mission is to investigate a target company provided by the user, scrape relevant web data, and produce a highly structured, data-driven intelligence report.

Required Report Structure:

Executive Summary: A concise, one-paragraph overview of the company.

Company Profile:
- Specialization: What is their exact niche, core technology, or primary service?
- Company Size: Number of employees (provide an exact number or estimated range based on scraped data).
- Headquarters / Key Locations: Primary operational bases.

Company Performance:
- Financials / Growth: Estimated revenue, funding rounds, market share, or notable growth metrics.
- Market Position: Who are their primary competitors?
- Core Products & Offerings: The specific products or services they sell and their target demographic.
- Recent Developments: Key news, leadership changes, or major events from the last 6-12 months.

Constraints: Ground your entire report strictly in the data provided. If a specific metric (like revenue or exact employee count) cannot be found, explicitly state 'Insufficient data found for this metric' rather than guessing. Output strictly in Markdown format."""

    user_prompt = f"Target Company: {company_name}\n\nScraped Data:\n{scraped_data}\n\nPlease generate the report based ONLY on the provided scraped data."

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }

    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.2
    }

    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        report = result["choices"][0]["message"]["content"]
        state["final_report"] = report
    except Exception as e:
        print(f"OpenAI API call failed: {e}")
        state["final_report"] = f"# Corporate Intelligence Report: {company_name}\n\nError generating report."

    return state


# Graph configuration
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
    import sys
    if len(sys.argv) > 1:
        company_name = sys.argv[1]
    else:
        company_name = input("Enter the target company name: ")

    initial_state = AgentState(
        company_name=company_name,
        search_urls=[],
        scraped_data="",
        final_report=""
    )

    print(f"Starting research for {company_name}...")

    # Execution using app.stream()
    final_state = None
    for output in app.stream(initial_state):
        for key, value in output.items():
            print(f"Node '{key}' completed.")
            final_state = value # capture state from the last node

    if final_state and "final_report" in final_state:
        print("\n=== FINAL REPORT ===\n")
        print(final_state["final_report"])
    else:
        print("\nFailed to generate report.")
