import os
import sys
import requests
from bs4 import BeautifulSoup
from typing import TypedDict, List
from langgraph.graph import StateGraph, END

# Define the state
class AgentState(TypedDict):
    company_name: str
    urls_to_scrape: List[str]
    raw_scraped_data: str
    final_report: str

def search_discovery(state: AgentState):
    company = state['company_name']
    tavily_key = os.environ.get("TAVILY_API_KEY")

    if not tavily_key:
        print("Error: TAVILY_API_KEY environment variable is missing.")
        sys.exit(1)

    queries = [
        f"{company} official website",
        f"{company} LinkedIn page",
        f"{company} recent financial business news"
    ]

    urls = []
    for query in queries:
        try:
            response = requests.post(
                "https://api.tavily.com/search",
                json={"api_key": tavily_key, "query": query, "search_depth": "basic"},
                timeout=10
            )
            response.raise_for_status()
            results = response.json().get("results", [])
            for r in results:
                urls.append(r["url"])
        except Exception as e:
            print(f"Error during search for query '{query}': {e}")

    # Deduplicate URLs
    unique_urls = list(set(urls))[:10]  # limit to top 10 unique URLs
    return {"urls_to_scrape": unique_urls}

def deep_scraper(state: AgentState):
    urls = state.get("urls_to_scrape", [])
    scraped_data = []

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    for url in urls:
        try:
            res = requests.get(url, headers=headers, timeout=15)
            if res.status_code == 200:
                soup = BeautifulSoup(res.text, 'html.parser')

                # Remove scripts and styles
                for script in soup(["script", "style", "nav", "footer", "header"]):
                    script.extract()

                text = soup.get_text(separator=' ', strip=True)

                # Keep up to 3000 characters per URL to fit in LLM context
                scraped_data.append(f"Source URL: {url}\nContent: {text[:3000]}")
        except Exception as e:
            print(f"Failed to scrape {url}: {e}")
            continue

    return {"raw_scraped_data": "\n\n---\n\n".join(scraped_data)}

def data_extraction_synthesis(state: AgentState):
    company = state["company_name"]
    raw_data = state.get("raw_scraped_data", "")
    openai_key = os.environ.get("OPENAI_API_KEY")

    if not openai_key:
        print("Error: OPENAI_API_KEY environment variable is missing.")
        sys.exit(1)

    system_prompt = """You are an elite Corporate Intelligence Researcher. Your mission is to investigate a target company provided by the user, scrape relevant web data, and produce a highly structured, data-driven intelligence report.

Your Execution Loop:
Plan: Identify the search queries needed to find the company's official site, corporate profiles (like LinkedIn or Crunchbase), and recent financial press.
Search & Scrape: Deploy your tools to extract raw text from these URLs. Look specifically for quantitative data.
Synthesize: Cross-reference the data, discard marketing fluff, and extract hard facts.

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

Constraints: Ground your entire report strictly in the scraped data provided below. If a specific metric (like revenue or exact employee count) cannot be found, explicitly state 'Insufficient data found for this metric' rather than guessing."""

    user_prompt = f"Target Company: {company}\n\nScraped Data:\n{raw_data}"

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {openai_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "gpt-4o",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.2
            },
            timeout=60
        )
        response.raise_for_status()
        report = response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        report = f"Failed to generate report: {e}\nResponse: {response.text if 'response' in locals() else 'No response'}"

    return {"final_report": report}

# Compile Graph
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
    if len(sys.argv) < 2:
        print('Usage: python corporate_intelligence_agent.py "[Company Name]"')
        sys.exit(1)

    company_name = sys.argv[1]
    print(f"Starting Corporate Intelligence Research for: {company_name}...\n")

    initial_state = {"company_name": company_name}

    try:
        result = app.invoke(initial_state)
        print("\n" + "="*50 + "\n")
        print(result["final_report"])
    except Exception as e:
        print(f"Pipeline execution failed: {e}")
