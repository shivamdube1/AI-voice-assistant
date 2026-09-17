import os
import sys
import json
import requests
from bs4 import BeautifulSoup
from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END

class AgentState(TypedDict):
    company_name: str
    search_urls: List[str]
    scraped_data: str
    final_report: str

def search_discovery(state: AgentState) -> AgentState:
    company_name = state["company_name"]
    tavily_api_key = os.environ.get("TAVILY_API_KEY")

    if not tavily_api_key:
        # Default to mock URLs if key is not found
        state["search_urls"] = [
            f"https://mock-company-site.com/{company_name.lower().replace(' ', '')}",
            f"https://mock-news.com/{company_name.lower().replace(' ', '')}"
        ]
        return state

    try:
        url = "https://api.tavily.com/search"
        headers = {"Content-Type": "application/json"}
        data = {
            "api_key": tavily_api_key,
            "query": f"{company_name} official website OR {company_name} financial business news",
            "search_depth": "basic",
            "include_answer": False,
            "max_results": 3
        }

        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()

        results = response.json().get("results", [])
        urls = [res["url"] for res in results]
        state["search_urls"] = urls
    except Exception as e:
        # Fallback to mock URLs on failure
        state["search_urls"] = [
            f"https://mock-company-site.com/{company_name.lower().replace(' ', '')}",
            f"https://mock-news.com/{company_name.lower().replace(' ', '')}"
        ]

    return state

def deep_scraper(state: AgentState) -> AgentState:
    urls = state.get("search_urls", [])
    scraped_texts = []

    for url in urls:
        if "linkedin.com" in url:
            continue

        try:
            # Attempt to fetch content
            if "mock-company-site" in url or "mock-news" in url:
                # Mock scrape
                scraped_texts.append(f"Mock scraped content for {url}: About Us, Careers, employees, headcount, revenue.")
            else:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    text = soup.get_text(separator=' ', strip=True)
                    # Simple heuristic: grab a chunk of text
                    scraped_texts.append(text[:2000])
        except Exception:
            continue

    state["scraped_data"] = "\n\n".join(scraped_texts)
    return state

def data_extraction_synthesis(state: AgentState) -> AgentState:
    company_name = state["company_name"]
    scraped_data = state.get("scraped_data", "")
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

    user_prompt = f"Target Company: {company_name}\n\nScraped Data:\n{scraped_data}\n\nGenerate the intelligence report based ONLY on the scraped data above."

    if not openai_api_key:
        state["final_report"] = f"# Corporate Intelligence Report: {company_name}\n\n[MOCK REPORT] Generated because OPENAI_API_KEY is missing.\n\nExecutive Summary: Mock summary.\n\nCompany Profile:\nSpecialization: Mock specialization\nCompany Size: Insufficient data found for this metric\nHeadquarters / Key Locations: Insufficient data found for this metric\n\nCompany Performance:\nFinancials / Growth: Insufficient data found for this metric\nMarket Position: Insufficient data found for this metric\nCore Products & Offerings: Insufficient data found for this metric\nRecent Developments: Insufficient data found for this metric"
        return state

    try:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai_api_key}"
        }
        data = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.2
        }

        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()

        result = response.json()
        report = result["choices"][0]["message"]["content"]
        state["final_report"] = report
    except Exception as e:
        state["final_report"] = f"Error generating report: {str(e)}"

    return state

# Graph Compilation Logic
workflow = StateGraph(AgentState)

workflow.add_node("search_discovery", search_discovery)
workflow.add_node("deep_scraper", deep_scraper)
workflow.add_node("data_extraction_synthesis", data_extraction_synthesis)

workflow.add_edge(START, "search_discovery")
workflow.add_edge("search_discovery", "deep_scraper")
workflow.add_edge("deep_scraper", "data_extraction_synthesis")
workflow.add_edge("data_extraction_synthesis", END)

app = workflow.compile()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python corporate_intelligence_agent.py '[Company Name]'")
        sys.exit(1)

    target_company = sys.argv[1]

    initial_state = {
        "company_name": target_company,
        "search_urls": [],
        "scraped_data": "",
        "final_report": ""
    }

    print(f"Starting analysis for: {target_company}\n")

    # Avoid double execution by capturing the final state directly from app.stream
    final_state = None
    for output in app.stream(initial_state):
        for key, value in output.items():
            print(f"Finished node: {key}")
            final_state = value

    if final_state and "final_report" in final_state:
        print("\n--- Final Report ---\n")
        print(final_state["final_report"])
    else:
        print("\nPipeline failed to generate a report.")
