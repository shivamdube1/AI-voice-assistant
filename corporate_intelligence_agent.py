import os
import sys
import requests
from bs4 import BeautifulSoup
from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END

class AgentState(TypedDict):
    company_name: str
    search_results: List[str]
    scraped_data: str
    final_report: str

def search_discovery(state: AgentState):
    company_name = state["company_name"]
    tavily_key = os.environ.get("TAVILY_API_KEY")
    if not tavily_key:
        print("Error: TAVILY_API_KEY environment variable is missing.")
        sys.exit(1)

    queries = [
        f"{company_name} official website",
        f"{company_name} LinkedIn profile",
        f"{company_name} recent financial business news"
    ]

    urls = []
    seen = set()
    url_endpoint = "https://api.tavily.com/search"

    print("Running Search & Discovery...")
    for query in queries:
        data = {
            "api_key": tavily_key,
            "query": query,
            "max_results": 3
        }
        try:
            resp = requests.post(url_endpoint, json=data)
            if resp.status_code == 200:
                res_json = resp.json()
                for r in res_json.get("results", []):
                    url = r["url"]
                    if url not in seen:
                        seen.add(url)
                        urls.append(url)
        except Exception as e:
            print(f"Error during Tavily search for '{query}': {e}")

    print(f"Found {len(urls)} URLs to scrape.")
    return {"search_results": urls}

def deep_scraper(state: AgentState):
    urls = state.get("search_results", [])
    all_text = ""

    print("Running Deep Scraper...")
    for url in urls:
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "html.parser")
                text = soup.get_text(separator=' ', strip=True)
                all_text += f"\n--- Source: {url} ---\n"
                all_text += text[:10000]  # Limiting per URL to avoid exceeding context limits
        except Exception as e:
            print(f"Failed to scrape {url}: {e}")

    print("Finished scraping.")
    return {"scraped_data": all_text}

def data_extraction_synthesis(state: AgentState):
    company_name = state["company_name"]
    scraped_data = state.get("scraped_data", "")

    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        print("Error: OPENAI_API_KEY environment variable is missing.")
        sys.exit(1)

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

    user_prompt = f"Target Company: {company_name}\n\nScraped Data:\n{scraped_data}\n\nPlease generate the report based on the provided constraints and structure."

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_key}"
    }
    data = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.2
    }

    print("Running Data Extraction & Synthesis...")
    try:
        resp = requests.post(url, headers=headers, json=data)
        resp.raise_for_status()
        report = resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error during OpenAI API call: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(e.response.text)
        report = "Error generating report."

    return {"final_report": report}

def main():
    if len(sys.argv) < 2:
        print('Usage: python corporate_intelligence_agent.py "[Company Name]"')
        sys.exit(1)

    company_name = sys.argv[1]

    workflow = StateGraph(AgentState)
    workflow.add_node("search_discovery", search_discovery)
    workflow.add_node("deep_scraper", deep_scraper)
    workflow.add_node("data_extraction_synthesis", data_extraction_synthesis)

    workflow.add_edge(START, "search_discovery")
    workflow.add_edge("search_discovery", "deep_scraper")
    workflow.add_edge("deep_scraper", "data_extraction_synthesis")
    workflow.add_edge("data_extraction_synthesis", END)

    app = workflow.compile()

    print(f"Starting investigation for: {company_name}...")
    final_state = app.invoke({"company_name": company_name})

    print("\n" + "="*50)
    print("FINAL REPORT")
    print("="*50 + "\n")
    print(final_state.get("final_report", "No report generated."))

if __name__ == "__main__":
    main()
