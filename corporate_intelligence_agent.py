import os
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
    company_name = state["company_name"]
    tavily_api_key = os.environ.get("TAVILY_API_KEY")

    if tavily_api_key:
        try:
            response = requests.post(
                "https://api.tavily.com/search",
                headers={"Content-Type": "application/json"},
                json={
                    "api_key": tavily_api_key,
                    "query": f"{company_name} official website OR LinkedIn OR financial news",
                    "search_depth": "basic",
                    "include_answer": False,
                    "include_raw_content": False,
                    "max_results": 5
                },
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            urls = [result["url"] for result in data.get("results", [])]
        except Exception as e:
            print(f"Error during Tavily search: {e}")
            urls = [f"https://mockurl.com/about_{company_name.lower()}"]
    else:
        urls = [f"https://mockurl.com/about_{company_name.lower()}"]

    state["search_urls"] = urls
    return state

def deep_scraper(state: AgentState) -> AgentState:
    urls = state["search_urls"]
    scraped_text_list = []

    for url in urls:
        if "linkedin.com" in url.lower():
            print(f"Skipping LinkedIn URL to avoid blocks: {url}")
            continue

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            # Remove scripts and styles
            for script in soup(["script", "style"]):
                script.extract()

            text = soup.get_text(separator=' ')
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)

            scraped_text_list.append(f"Source: {url}\n{text[:5000]}") # Limit text per source
        except Exception as e:
            print(f"Error scraping {url}: {e}")

    state["scraped_data"] = "\n\n".join(scraped_text_list)
    return state

def data_extraction_synthesis(state: AgentState) -> AgentState:
    company_name = state["company_name"]
    scraped_data = state["scraped_data"]
    openai_api_key = os.environ.get("OPENAI_API_KEY")

    prompt = f"""
You are an elite Corporate Intelligence Researcher. Your mission is to investigate {company_name},
using the scraped web data provided below, and produce a highly structured, data-driven intelligence report.

Synthesize the data, discard marketing fluff, and extract hard facts.

Required Report Structure:
# Executive Summary
A concise, one-paragraph overview of the company.

# Company Profile
- Specialization: What is their exact niche, core technology, or primary service?
- Company Size: Number of employees (provide an exact number or estimated range based on scraped data).
- Headquarters / Key Locations: Primary operational bases.

# Company Performance
- Financials / Growth: Estimated revenue, funding rounds, market share, or notable growth metrics.
- Market Position: Who are their primary competitors?
- Core Products & Offerings: The specific products or services they sell and their target demographic.
- Recent Developments: Key news, leadership changes, or major events from the last 6-12 months.

Constraints:
Ground your entire report strictly in the data provided below.
If a specific metric (like revenue or exact employee count) cannot be found in the data,
explicitly state 'Insufficient data found for this metric' rather than guessing.

Scraped Data:
{scraped_data}
"""

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
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2
                },
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            report = data["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"Error during OpenAI synthesis: {e}")
            report = f"# Report for {company_name}\nError generating report."
    else:
        report = f"""# Executive Summary
Mock summary for {company_name}.

# Company Profile
- Specialization: Insufficient data found for this metric
- Company Size: Insufficient data found for this metric
- Headquarters / Key Locations: Insufficient data found for this metric

# Company Performance
- Financials / Growth: Insufficient data found for this metric
- Market Position: Insufficient data found for this metric
- Core Products & Offerings: Insufficient data found for this metric
- Recent Developments: Insufficient data found for this metric"""

    state["final_report"] = report
    return state

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

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print('Usage: python corporate_intelligence_agent.py "[Company Name]"')
        sys.exit(1)

    company_name = sys.argv[1]
    print(f"Starting research on {company_name}...")

    graph = build_graph()

    initial_state = {
        "company_name": company_name,
        "search_urls": [],
        "scraped_data": "",
        "final_report": ""
    }

    result = graph.invoke(initial_state)

    print("\n" + "="*50)
    print("FINAL REPORT:")
    print("="*50 + "\n")
    print(result.get("final_report", "Report generation failed."))
