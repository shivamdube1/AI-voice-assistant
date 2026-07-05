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
    print(f"--- Search & Discovery Node: Finding info for {state['company_name']} ---")
    tavily_api_key = os.environ.get("TAVILY_API_KEY")
    if not tavily_api_key:
        print("Warning: TAVILY_API_KEY not found. Returning empty search_urls.")
        return {"search_urls": []}

    query = f"{state['company_name']} official website OR LinkedIn OR recent financial business news"

    url = "https://api.tavily.com/search"
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
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        urls = [result.get('url') for result in data.get('results', []) if result.get('url')]
        return {"search_urls": urls}
    except Exception as e:
        print(f"Error during Tavily search: {e}")
        return {"search_urls": []}

def deep_scraper(state: AgentState) -> AgentState:
    print(f"--- Deep Scraper Node: Scraping {len(state.get('search_urls', []))} URLs ---")
    urls = state.get("search_urls", [])
    scraped_text_list = []

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    for url in urls:
        try:
            print(f"Scraping {url}...")
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # Remove script and style elements
            for script_or_style in soup(['script', 'style']):
                script_or_style.extract()

            text = soup.get_text(separator=' ')
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)

            # Limit the text from each URL to avoid huge context sizes
            scraped_text_list.append(f"Source: {url}\nContent: {text[:2000]}")
        except Exception as e:
            print(f"Failed to scrape {url}: {e}")

    combined_data = "\n\n".join(scraped_text_list)
    return {"scraped_data": combined_data}

def data_extraction_synthesis(state: AgentState) -> AgentState:
    print("--- Data Extraction & Synthesis Node: Generating report ---")
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        print("Warning: OPENAI_API_KEY not found. Returning empty report.")
        return {"final_report": "Error: OPENAI_API_KEY is missing."}

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

Constraints: Ground your entire report strictly in the data you scrape. If a specific metric (like revenue or exact employee count) cannot be found, explicitly state 'Insufficient data found for this metric' rather than guessing."""

    user_prompt = f"Company: {state['company_name']}\n\nScraped Data:\n{state.get('scraped_data', '')}"

    url = "https://api.openai.com/v1/chat/completions"
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
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        report = data['choices'][0]['message']['content']
        return {"final_report": report}
    except Exception as e:
        print(f"Error during OpenAI API call: {e}")
        return {"final_report": f"Error generating report: {e}"}

def build_graph():
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

    graph = build_graph()

    initial_state = {
        "company_name": company_name,
        "search_urls": [],
        "scraped_data": "",
        "final_report": ""
    }

    result = graph.invoke(initial_state)

    print("\n" + "="*80)
    print("FINAL REPORT:")
    print("="*80)
    print(result.get("final_report", "No report generated."))
