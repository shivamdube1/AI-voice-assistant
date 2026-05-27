import os
import sys
import json
import requests
from bs4 import BeautifulSoup
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define the State
class AgentState(TypedDict):
    company_name: str
    search_queries: List[str]
    search_results: List[Dict[str, str]]  # list of dicts with 'url' and 'content' snippet
    scraped_data: str  # aggregated text from scraping
    report: str


def search_discovery(state: AgentState) -> dict:
    company_name = state.get("company_name", "")
    print(f"[*] Discovering information for: {company_name}")

    # Formulate queries
    queries = [
        f"{company_name} official website",
        f"{company_name} LinkedIn company profile",
        f"{company_name} recent financial business news"
    ]

    search_results = []
    tavily_api_key = os.environ.get("TAVILY_API_KEY", "")

    if not tavily_api_key:
        print("Warning: TAVILY_API_KEY not found in environment.")
        return {"search_queries": queries, "search_results": []}

    url = "https://api.tavily.com/search"
    headers = {
        "Content-Type": "application/json"
    }

    # Collect results
    for query in queries:
        payload = {
            "api_key": tavily_api_key,
            "query": query,
            "search_depth": "basic",
            "max_results": 2
        }
        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            for res in data.get("results", []):
                search_results.append({
                    "url": res.get("url", ""),
                    "content": res.get("content", "")
                })
        except Exception as e:
            print(f"Error searching for '{query}': {e}")

    # Remove duplicates based on URL
    unique_results = []
    seen_urls = set()
    for res in search_results:
        if res["url"] not in seen_urls:
            seen_urls.add(res["url"])
            unique_results.append(res)

    return {"search_queries": queries, "search_results": unique_results}


def deep_scraper(state: AgentState) -> dict:
    search_results = state.get("search_results", [])
    print(f"[*] Deep scraping {len(search_results)} URLs...")

    scraped_texts = []

    for res in search_results:
        url = res.get("url", "")
        snippet = res.get("content", "")

        scraped_texts.append(f"Source: {url}\nSnippet: {snippet}")

        try:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
            response = requests.get(url, headers=headers, timeout=5)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                for script in soup(["script", "style"]):
                    script.extract()
                text = soup.get_text(separator=' ')
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = '\n'.join(chunk for chunk in chunks if chunk)

                # Truncate text to avoid massive context
                scraped_texts.append(f"Scraped Content:\n{text[:2000]}...\n")
        except Exception as e:
            print(f"Failed to scrape {url}: {e}")

    scraped_data = "\n".join(scraped_texts)

    return {"scraped_data": scraped_data[:30000]}


def data_extraction_synthesis(state: AgentState) -> dict:
    company_name = state.get("company_name", "")
    scraped_data = state.get("scraped_data", "")
    print(f"[*] Synthesizing data for {company_name}...")

    openai_api_key = os.environ.get("OPENAI_API_KEY", "")
    if not openai_api_key:
        print("Warning: OPENAI_API_KEY not found in environment.")
        return {"report": "Error: OPENAI_API_KEY missing."}

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

    user_prompt = f"Target Company: {company_name}\n\nScraped Data:\n{scraped_data}\n\nPlease generate the corporate intelligence report based on the provided scraped data."

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.2
    }

    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        report = result["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error during synthesis: {e}")
        report = f"Error generating report: {e}"

    return {"report": report}


def compile_agent() -> StateGraph:
    workflow = StateGraph(AgentState)

    workflow.add_node("search_discovery", search_discovery)
    workflow.add_node("deep_scraper", deep_scraper)
    workflow.add_node("data_extraction_synthesis", data_extraction_synthesis)

    workflow.set_entry_point("search_discovery")
    workflow.add_edge("search_discovery", "deep_scraper")
    workflow.add_edge("deep_scraper", "data_extraction_synthesis")
    workflow.add_edge("data_extraction_synthesis", END)

    return workflow.compile()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python corporate_intelligence_agent.py "[Company Name]"')
        sys.exit(1)

    target_company = sys.argv[1]

    agent = compile_agent()

    initial_state = {"company_name": target_company}

    print("==================================================")
    print(f"Starting Corporate Intelligence Agent for: {target_company}")
    print("==================================================")

    try:
        final_state = agent.invoke(initial_state)
        print("\n================ FINAL REPORT ================\n")
        print(final_state.get("report", "No report generated."))
    except Exception as e:
        print(f"Agent execution failed: {e}")
