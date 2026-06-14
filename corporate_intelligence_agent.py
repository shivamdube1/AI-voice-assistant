import os
import sys
import json
import requests
from bs4 import BeautifulSoup
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END

class AgentState(TypedDict):
    company_name: str
    search_results: List[Dict[str, str]]
    scraped_data: str
    final_report: str

def search_discovery(state: AgentState) -> Dict[str, Any]:
    company_name = state["company_name"]
    tavily_api_key = os.environ.get("TAVILY_API_KEY")
    if not tavily_api_key:
        print("Warning: TAVILY_API_KEY not set")
        return {"search_results": []}

    queries = [
        f"{company_name} official website",
        f"{company_name} LinkedIn profile",
        f"{company_name} recent financial business news revenue"
    ]

    all_results = []

    url = "https://api.tavily.com/search"
    headers = {
        "Content-Type": "application/json"
    }

    for query in queries:
        payload = {
            "api_key": tavily_api_key,
            "query": query,
            "search_depth": "basic",
            "include_answer": False,
            "include_images": False,
            "include_raw_content": False,
            "max_results": 3
        }

        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            for result in data.get("results", []):
                all_results.append({
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "content": result.get("content", "")
                })
        except Exception as e:
            print(f"Error during Tavily search for query '{query}': {e}")

    # Deduplicate URLs
    seen_urls = set()
    unique_results = []
    for res in all_results:
        if res["url"] not in seen_urls:
            unique_results.append(res)
            seen_urls.add(res["url"])

    return {"search_results": unique_results}

def deep_scraper(state: AgentState) -> Dict[str, Any]:
    search_results = state.get("search_results", [])
    scraped_data_parts = []

    # We will limit the number of URLs to scrape to avoid taking too long or making too many requests
    urls_to_scrape = [res["url"] for res in search_results[:5]]

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    for url in urls_to_scrape:
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')

                # Extract text
                # Kill all script and style elements
                for script in soup(["script", "style"]):
                    script.extract()

                text = soup.get_text(separator=' ', strip=True)

                # break into lines and remove leading and trailing space on each
                lines = (line.strip() for line in text.splitlines())
                # break multi-headlines into a line each
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                # drop blank lines
                text = '\n'.join(chunk for chunk in chunks if chunk)

                # Truncate text to avoid blowing up context window
                truncated_text = text[:3000]

                scraped_data_parts.append(f"URL: {url}\nContent:\n{truncated_text}\n---")
        except Exception as e:
            print(f"Error scraping {url}: {e}")

    return {"scraped_data": "\n".join(scraped_data_parts)}

def data_extraction_synthesis(state: AgentState) -> Dict[str, Any]:
    company_name = state["company_name"]
    scraped_data = state.get("scraped_data", "")
    search_context = "\n".join([f"{r['title']} - {r['content']}" for r in state.get("search_results", [])])

    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        print("Warning: OPENAI_API_KEY not set")
        return {"final_report": "OPENAI_API_KEY not set. Cannot generate report."}

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

    user_prompt = f"""Target Company: {company_name}

Here is the search context:
{search_context[:2000]}

Here is the scraped data:
{scraped_data[:10000]}

Generate the final Markdown report based strictly on the above data."""

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
        report = data["choices"][0]["message"]["content"]
        return {"final_report": report}
    except Exception as e:
        print(f"Error during OpenAI API call: {e}")
        if 'response' in locals() and hasattr(response, 'text'):
            print(f"Response: {response.text}")
        return {"final_report": "Error generating report via OpenAI."}

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
    if len(sys.argv) < 2:
        print('Usage: python corporate_intelligence_agent.py "[Company Name]"')
        sys.exit(1)

    company_name = sys.argv[1]

    app = build_graph()

    print(f"Starting research on {company_name}...")

    initial_state = {"company_name": company_name}

    final_state = app.invoke(initial_state)

    print("\n" + "="*50 + "\n")
    print(final_state.get("final_report", "No report generated."))
    print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main()
