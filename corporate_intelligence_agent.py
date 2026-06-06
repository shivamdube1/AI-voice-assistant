import os
import requests
from bs4 import BeautifulSoup
import argparse
from typing import TypedDict, List
from langgraph.graph import StateGraph, END

class AgentState(TypedDict):
    company_name: str
    urls_to_scrape: List[str]
    scraped_data: str
    final_report: str

def search_discovery(state: AgentState):
    company_name = state["company_name"]
    tavily_api_key = os.environ.get("TAVILY_API_KEY")
    if not tavily_api_key:
        raise ValueError("TAVILY_API_KEY environment variable is missing.")

    queries = [
        f"{company_name} official website",
        f"{company_name} LinkedIn page",
        f"{company_name} recent financial business news"
    ]

    urls = []
    scraped_snippets = []

    for query in queries:
        payload = {
            "api_key": tavily_api_key,
            "query": query,
            "max_results": 3,
            "search_depth": "basic"
        }
        response = requests.post("https://api.tavily.com/search", json=payload)
        if response.status_code == 200:
            data = response.json()
            for result in data.get("results", []):
                url = result.get("url")
                if url and url not in urls:
                    urls.append(url)
                scraped_snippets.append(f"Source: {url}\nSnippet: {result.get('content')}")
        else:
            print(f"Tavily API error for query '{query}': {response.text}")

    return {"urls_to_scrape": urls, "scraped_data": "\n\n".join(scraped_snippets) + "\n\n"}

def deep_scraper(state: AgentState):
    urls = state.get("urls_to_scrape", [])
    scraped_data = state.get("scraped_data", "")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    scraped_texts = []
    # Limit to top 5 urls to avoid excessively long prompts
    for url in urls[:5]:
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.extract()
                text = soup.get_text(separator=' ', strip=True)
                # Truncate text to avoid massive token usage
                scraped_texts.append(f"URL: {url}\nContent: {text[:3000]}")
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            continue

    new_data = scraped_data + "\n\n".join(scraped_texts)
    return {"scraped_data": new_data}

def data_extraction_synthesis(state: AgentState):
    company_name = state["company_name"]
    scraped_data = state.get("scraped_data", "")
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable is missing.")

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

    user_prompt = f"Target Company: {company_name}\n\nScraped Data:\n{scraped_data}\n\nPlease generate the required report based strictly on the data above."

    payload = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.2
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers)
    if response.status_code != 200:
        raise ValueError(f"OpenAI API Error: {response.text}")

    data = response.json()
    final_report = data["choices"][0]["message"]["content"]

    return {"final_report": final_report}

def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("search_discovery", search_discovery)
    workflow.add_node("deep_scraper", deep_scraper)
    workflow.add_node("data_extraction_synthesis", data_extraction_synthesis)

    workflow.set_entry_point("search_discovery")
    workflow.add_edge("search_discovery", "deep_scraper")
    workflow.add_edge("deep_scraper", "data_extraction_synthesis")
    workflow.add_edge("data_extraction_synthesis", END)

    return workflow.compile()

def main():
    parser = argparse.ArgumentParser(description="Autonomous Corporate Intelligence Agent")
    parser.add_argument("company_name", type=str, help="The name of the target company")
    args = parser.parse_args()

    graph = build_graph()

    initial_state = {"company_name": args.company_name, "urls_to_scrape": [], "scraped_data": "", "final_report": ""}

    print(f"Starting research on: {args.company_name}...")
    result = graph.invoke(initial_state)

    print("\n" + "="*50 + "\n")
    print(result.get("final_report", "No report generated."))
    print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main()
