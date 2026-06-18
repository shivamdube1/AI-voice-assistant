import os
import sys
import requests
from bs4 import BeautifulSoup
from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END

class AgentState(TypedDict):
    company_name: str
    urls_to_scrape: List[str]
    scraped_data: str
    final_report: str

def search_discovery(state: AgentState) -> AgentState:
    company_name = state["company_name"]
    tavily_api_key = os.environ.get("TAVILY_API_KEY")

    if not tavily_api_key:
        raise ValueError("TAVILY_API_KEY environment variable is not set.")

    query = f"{company_name} official website LinkedIn recent financial business news"

    payload = {
        "api_key": tavily_api_key,
        "query": query,
        "search_depth": "advanced",
        "include_answer": False,
        "max_results": 5
    }

    response = requests.post("https://api.tavily.com/search", json=payload)
    response.raise_for_status()
    data = response.json()

    urls = [result["url"] for result in data.get("results", [])]

    return {"urls_to_scrape": urls}

def deep_scraper(state: AgentState) -> AgentState:
    urls = state["urls_to_scrape"]
    scraped_data = []

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    for url in urls:
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                for script in soup(["script", "style"]):
                    script.extract()
                text = soup.get_text(separator=' ', strip=True)
                scraped_data.append(f"Source: {url}\nContent: {text[:5000]}")
        except Exception as e:
            pass

    return {"scraped_data": "\n\n---\n\n".join(scraped_data)}

def data_extraction_synthesis(state: AgentState) -> AgentState:
    company_name = state["company_name"]
    scraped_data = state["scraped_data"]
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

    user_prompt = f"Target Company: {company_name}\n\nHere is the scraped data:\n{scraped_data}\n\nPlease generate the corporate intelligence report based on this data according to your instructions."

    payload = {
        "model": "gpt-4-turbo",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.2
    }

    headers = {
        "Authorization": f"Bearer {openai_api_key}",
        "Content-Type": "application/json"
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers)
    response.raise_for_status()
    data = response.json()

    final_report = data["choices"][0]["message"]["content"]

    return {"final_report": final_report}

def create_agent():
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

    agent = create_agent()

    print(f"Investigating {company_name}...")

    initial_state = {
        "company_name": company_name,
        "urls_to_scrape": [],
        "scraped_data": "",
        "final_report": ""
    }

    try:
        final_state = agent.invoke(initial_state)
        print("\n=== Corporate Intelligence Report ===\n")
        print(final_state["final_report"])
    except Exception as e:
        print(f"Error during execution: {e}")
