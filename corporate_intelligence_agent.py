import os
import sys
import json
import requests
from typing import TypedDict, List
from bs4 import BeautifulSoup
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
        return {"search_urls": [f"https://mock.url/{company_name.lower().replace(' ', '')}"]}

    query = f"official website, linkedin page, recent financial business news for {company_name}"

    headers = {"Content-Type": "application/json"}
    payload = {
        "api_key": tavily_api_key,
        "query": query,
        "search_depth": "basic",
        "include_answer": False,
        "include_images": False,
        "include_raw_content": False,
        "max_results": 5,
        "include_domains": [],
        "exclude_domains": []
    }

    try:
        response = requests.post("https://api.tavily.com/search", json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        urls = [result.get("url") for result in data.get("results", []) if result.get("url")]
        return {"search_urls": urls}
    except Exception as e:
        print(f"Error during search: {e}")
        return {"search_urls": []}

def deep_scraper(state: AgentState) -> AgentState:
    urls = state.get("search_urls", [])
    scraped_text = []

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    for url in urls:
        if "linkedin.com" in url.lower():
            continue
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                text = soup.get_text(separator=' ', strip=True)
                scraped_text.append(f"Source: {url}\n{text[:5000]}")  # Limit to 5000 chars per source to avoid huge context
        except Exception as e:
            print(f"Error scraping {url}: {e}")

    return {"scraped_data": "\n\n".join(scraped_text)}

def data_extraction_synthesis(state: AgentState) -> AgentState:
    company_name = state["company_name"]
    scraped_data = state.get("scraped_data", "")
    openai_api_key = os.environ.get("OPENAI_API_KEY")

    system_prompt = (
        "You are an elite Corporate Intelligence Researcher. Your mission is to investigate a target company provided by the user, "
        "scrape relevant web data, and produce a highly structured, data-driven intelligence report.\n\n"
        "Required Report Structure:\n"
        "Executive Summary: A concise, one-paragraph overview of the company.\n"
        "Company Profile:\n"
        "Specialization: What is their exact niche, core technology, or primary service?\n"
        "Company Size: Number of employees (provide an exact number or estimated range based on scraped data).\n"
        "Headquarters / Key Locations: Primary operational bases.\n"
        "Company Performance:\n"
        "Financials / Growth: Estimated revenue, funding rounds, market share, or notable growth metrics.\n"
        "Market Position: Who are their primary competitors?\n"
        "Core Products & Offerings: The specific products or services they sell and their target demographic.\n"
        "Recent Developments: Key news, leadership changes, or major events from the last 6-12 months.\n\n"
        "Constraints: Ground your entire report strictly in the data you scrape. If a specific metric (like revenue or exact employee count) "
        "cannot be found, explicitly state 'Insufficient data found for this metric' rather than guessing."
    )

    user_prompt = f"Target Company: {company_name}\n\nScraped Data:\n{scraped_data}\n\nPlease generate the report based ONLY on this scraped data."

    if not openai_api_key:
        fallback_report = f"# Intelligence Report for {company_name}\n\n" \
                          f"**Executive Summary:**\nInsufficient data found for this metric.\n\n" \
                          f"**Company Profile:**\n" \
                          f"- Specialization: Insufficient data found for this metric.\n" \
                          f"- Company Size: Insufficient data found for this metric.\n" \
                          f"- Headquarters / Key Locations: Insufficient data found for this metric.\n\n" \
                          f"**Company Performance:**\n" \
                          f"- Financials / Growth: Insufficient data found for this metric.\n" \
                          f"- Market Position: Insufficient data found for this metric.\n" \
                          f"- Core Products & Offerings: Insufficient data found for this metric.\n" \
                          f"- Recent Developments: Insufficient data found for this metric.\n"
        return {"final_report": fallback_report}

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
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        report = result["choices"][0]["message"]["content"]
        return {"final_report": report}
    except Exception as e:
        print(f"Error during synthesis: {e}")
        return {"final_report": "Error generating report due to API failure."}

def compile_graph():
    graph = StateGraph(AgentState)
    graph.add_node("search_discovery", search_discovery)
    graph.add_node("deep_scraper", deep_scraper)
    graph.add_node("data_extraction_synthesis", data_extraction_synthesis)

    graph.add_edge(START, "search_discovery")
    graph.add_edge("search_discovery", "deep_scraper")
    graph.add_edge("deep_scraper", "data_extraction_synthesis")
    graph.add_edge("data_extraction_synthesis", END)

    return graph.compile()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python corporate_intelligence_agent.py "[Company Name]"')
        sys.exit(1)

    company = sys.argv[1]
    agent = compile_graph()

    print(f"Starting research on {company}...")
    initial_state = {"company_name": company}

    result = agent.invoke(initial_state)
    print("\n\n" + "="*50 + "\n\n")
    print(result.get("final_report", "No report generated."))
