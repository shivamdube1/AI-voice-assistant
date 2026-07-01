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
    tavily_api_key = os.environ.get("TAVILY_API_KEY", "")

    query = f"{company_name} official website OR LinkedIn OR recent financial business news"

    urls = []
    if tavily_api_key:
        try:
            response = requests.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": tavily_api_key,
                    "query": query,
                    "search_depth": "advanced",
                    "max_results": 5
                },
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            results = response.json().get("results", [])
            urls = [res["url"] for res in results]
        except Exception as e:
            print(f"Error searching Tavily: {e}")
    else:
        print("Warning: TAVILY_API_KEY not set.")

    return {
        "company_name": state["company_name"],
        "search_urls": urls,
        "scraped_data": state.get("scraped_data", ""),
        "final_report": state.get("final_report", "")
    }

def deep_scraper(state: AgentState) -> AgentState:
    urls = state.get("search_urls", [])
    scraped_text_chunks = []

    for url in urls:
        try:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) CorporateIntelligenceAgent/1.0"}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            text = soup.get_text(separator=" ", strip=True)
            scraped_text_chunks.append(f"Source: {url}\nContent: {text[:5000]}")
        except Exception as e:
            print(f"Error scraping {url}: {e}")

    scraped_data = "\n\n".join(scraped_text_chunks)
    return {
        "company_name": state["company_name"],
        "search_urls": state["search_urls"],
        "scraped_data": scraped_data,
        "final_report": state.get("final_report", "")
    }

def data_extraction_synthesis(state: AgentState) -> AgentState:
    company_name = state["company_name"]
    scraped_data = state.get("scraped_data", "")
    openai_api_key = os.environ.get("OPENAI_API_KEY", "")

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

    user_prompt = f"Target Company: {company_name}\n\nScraped Data:\n{scraped_data}\n\nPlease generate the intelligence report based on the provided scraped data."

    final_report = "Insufficient data found for this metric"
    if openai_api_key:
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
            print(f"Error calling OpenAI API: {e}")
            final_report = f"Error generating report: {e}"
    else:
        print("Warning: OPENAI_API_KEY not set.")
        final_report = "OPENAI_API_KEY not set. Cannot generate report."

    return {
        "company_name": state["company_name"],
        "search_urls": state["search_urls"],
        "scraped_data": state["scraped_data"],
        "final_report": final_report
    }

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
        print("Usage: python corporate_intelligence_agent.py \"[Company Name]\"")
        sys.exit(1)

    target_company = sys.argv[1]

    initial_state: AgentState = {
        "company_name": target_company,
        "search_urls": [],
        "scraped_data": "",
        "final_report": ""
    }

    print(f"Running Corporate Intelligence Agent for: {target_company}")
    result_state = app.invoke(initial_state)

    print("\n" + "="*50)
    print("FINAL REPORT")
    print("="*50 + "\n")
    print(result_state.get("final_report", "No report generated."))
