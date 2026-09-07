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
    """
    Search & Discovery Node: Use a search tool (Tavily) to find the company's official website,
    LinkedIn page, and recent financial/business news.
    """
    company_name = state["company_name"]
    tavily_api_key = os.environ.get("TAVILY_API_KEY")

    if not tavily_api_key:
        print("TAVILY_API_KEY not found. Using mock URLs.")
        urls = [
            f"https://www.example.com/{company_name.lower().replace(' ', '')}",
            f"https://www.linkedin.com/company/{company_name.lower().replace(' ', '')}",
            f"https://news.example.com/company/{company_name.lower().replace(' ', '')}"
        ]
        return {"search_urls": urls}

    query = f"official website, LinkedIn page, and recent financial business news for {company_name}"

    headers = {
        "Content-Type": "application/json"
    }

    payload = {
        "api_key": tavily_api_key,
        "query": query,
        "search_depth": "basic",
        "include_answer": False,
        "include_images": False,
        "include_raw_content": False,
        "max_results": 5
    }

    try:
        response = requests.post("https://api.tavily.com/search", headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

        urls = [result["url"] for result in data.get("results", [])]

        # Ensure we have at least some URLs
        if not urls:
            print("No URLs found via Tavily API. Using mock URLs.")
            urls = [
                f"https://www.example.com/{company_name.lower().replace(' ', '')}",
                f"https://www.linkedin.com/company/{company_name.lower().replace(' ', '')}",
                f"https://news.example.com/company/{company_name.lower().replace(' ', '')}"
            ]

        return {"search_urls": urls}
    except Exception as e:
        print(f"Error calling Tavily API: {e}. Using mock URLs.")
        urls = [
            f"https://www.example.com/{company_name.lower().replace(' ', '')}",
            f"https://www.linkedin.com/company/{company_name.lower().replace(' ', '')}",
            f"https://news.example.com/company/{company_name.lower().replace(' ', '')}"
        ]
        return {"search_urls": urls}

def deep_scraper(state: AgentState) -> AgentState:
    """
    Deep Scraper Node: Extract text from URLs.
    Explicitly skips linkedin.com URLs to avoid request blocks.
    """
    search_urls = state.get("search_urls", [])
    scraped_texts = []

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    for url in search_urls:
        if "linkedin.com" in url.lower():
            print(f"Skipping linkedin.com URL: {url}")
            continue

        print(f"Scraping: {url}")
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract text and clean it up a bit
            text = soup.get_text(separator=' ', strip=True)
            scraped_texts.append(f"Source: {url}\n{text[:5000]}") # Limit text per URL to avoid huge contexts
        except Exception as e:
            print(f"Failed to scrape {url}: {e}")

    combined_text = "\n\n---\n\n".join(scraped_texts)

    if not combined_text:
        combined_text = "No data successfully scraped."

    return {"scraped_data": combined_text}

def data_extraction_synthesis(state: AgentState) -> AgentState:
    """
    Data Extraction & Synthesis Node: Pass raw data through an LLM to extract metrics
    and generate the final Markdown report.
    """
    scraped_data = state.get("scraped_data", "")
    company_name = state.get("company_name", "")
    openai_api_key = os.environ.get("OPENAI_API_KEY")

    if not openai_api_key:
        print("OPENAI_API_KEY not found. Returning a mock report.")
        mock_report = f"""# Corporate Intelligence Report: {company_name}

## Executive Summary
This is a mock report for {company_name} due to missing API keys.

## Company Profile
**Specialization:** Mock tech
**Company Size:** Insufficient data found for this metric
**Headquarters / Key Locations:** Unknown

## Company Performance
**Financials / Growth:** Insufficient data found for this metric
**Market Position:** Unknown competitors
**Core Products & Offerings:** Mock solutions
**Recent Developments:** No recent news
"""
        return {"final_report": mock_report}

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

    user_prompt = f"Target Company: {company_name}\n\nScraped Data:\n{scraped_data}\n\nGenerate the intelligence report."

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }

    payload = {
        "model": "gpt-4o",  # Can adjust model as needed
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
        return {"final_report": report}
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return {"final_report": f"Error generating report: {e}"}

# Graph Construction
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

    if len(sys.argv) < 2:
        print("Usage: python corporate_intelligence_agent.py \"[Company Name]\"")
        sys.exit(1)

    company_input = sys.argv[1]

    initial_state = {
        "company_name": company_input,
        "search_urls": [],
        "scraped_data": "",
        "final_report": ""
    }

    print(f"Starting analysis for: {company_input}...\n")

    final_state = initial_state
    for s in app.stream(initial_state, stream_mode="values"):
        final_state = s

    print("\n--- FINAL REPORT ---\n")
    print(final_state.get("final_report", "No report generated."))
