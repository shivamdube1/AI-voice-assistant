import os
import sys
import json
import requests
from bs4 import BeautifulSoup
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, START, END

class AgentState(TypedDict):
    company_name: str
    search_results: List[Dict[str, Any]]
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
        print("TAVILY_API_KEY is missing")
        return {"search_results": []}

    query = f"{company_name} official website, LinkedIn, recent financial news"

    url = "https://api.tavily.com/search"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "api_key": tavily_api_key,
        "query": query,
        "include_answer": False,
        "include_raw_content": False,
        "max_results": 5
    }

    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        results = response.json().get("results", [])
        return {"search_results": results}
    else:
        print(f"Error querying Tavily API: {response.text}")
        return {"search_results": []}

def deep_scraper(state: AgentState) -> AgentState:
    """
    Deep Scraper Node: Extract text from the official website (About Us, Careers) and external
    aggregators to specifically hunt for employee headcounts, performance metrics (revenue, growth),
    and core specializations.
    """
    search_results = state.get("search_results", [])
    scraped_texts = []

    timeout = 10
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    for result in search_results:
        url = result.get("url")
        if not url:
            continue
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, 'html.parser')
                for script in soup(["script", "style"]):
                    script.extract()
                text = soup.get_text(separator=' ')
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = ' '.join(chunk for chunk in chunks if chunk)

                scraped_texts.append(f"Source: {url}\nContent: {text[:5000]}")
            else:
                snippet = result.get("content", "")
                scraped_texts.append(f"Source: {url} (Snippet)\nContent: {snippet}")
        except Exception as e:
            snippet = result.get("content", "")
            scraped_texts.append(f"Source: {url} (Snippet)\nContent: {snippet}")

    combined_data = "\n\n".join(scraped_texts)
    return {"scraped_data": combined_data}

def data_extraction_synthesis(state: AgentState) -> AgentState:
    """
    Data Extraction & Synthesis Node: Pass the raw data through an LLM to extract the targeted
    business metrics, filter out noise, and generate a final Markdown report.
    """
    company_name = state["company_name"]
    scraped_data = state.get("scraped_data", "")

    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        print("OPENAI_API_KEY is missing")
        return {"final_report": "Error: OPENAI_API_KEY not found."}

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

    user_message = f"Company Name: {company_name}\n\nScraped Data:\n{scraped_data[:100000]}\n\nGenerate the report based ONLY on the provided scraped data."

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }
    data = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        "temperature": 0.2
    }

    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        report = response.json()["choices"][0]["message"]["content"]
        return {"final_report": report}
    else:
        print(f"Error calling OpenAI API: {response.text}")
        return {"final_report": "Error generating report."}

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("search_discovery", search_discovery)
    graph.add_node("deep_scraper", deep_scraper)
    graph.add_node("data_extraction_synthesis", data_extraction_synthesis)

    graph.add_edge(START, "search_discovery")
    graph.add_edge("search_discovery", "deep_scraper")
    graph.add_edge("deep_scraper", "data_extraction_synthesis")
    graph.add_edge("data_extraction_synthesis", END)

    return graph.compile()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python corporate_intelligence_agent.py \"[Company Name]\"")
        sys.exit(1)

    target_company = sys.argv[1]

    print(f"Starting Corporate Intelligence Agent for: {target_company}")

    app = build_graph()

    initial_state = {
        "company_name": target_company,
        "search_results": [],
        "scraped_data": "",
        "final_report": ""
    }

    result = app.invoke(initial_state)
    print("\n================ REPORT ================\n")
    print(result.get("final_report", "No report generated."))
    print("\n========================================\n")
