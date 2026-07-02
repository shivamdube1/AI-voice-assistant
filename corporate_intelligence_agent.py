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
    Use a search tool (Tavily) to find the company's official website,
    LinkedIn page, and recent financial/business news.
    """
    company_name = state["company_name"]
    tavily_api_key = os.environ.get("TAVILY_API_KEY")

    if not tavily_api_key:
        print("Warning: TAVILY_API_KEY not found. Skipping real search.")
        return {"search_urls": []}

    query = f"{company_name} official website OR LinkedIn OR recent financial business news"

    try:
        response = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": tavily_api_key,
                "query": query,
                "search_depth": "advanced",
                "include_images": False,
                "include_answers": False,
                "max_results": 5
            }
        )
        response.raise_for_status()
        data = response.json()
        urls = [result.get("url") for result in data.get("results", []) if result.get("url")]
        return {"search_urls": urls}
    except Exception as e:
        print(f"Error during search_discovery: {e}")
        return {"search_urls": []}

def deep_scraper(state: AgentState) -> AgentState:
    """
    Extract text from the official website (About Us, Careers) and external aggregators
    to specifically hunt for employee headcounts, performance metrics (revenue, growth),
    and core specializations.
    """
    urls = state.get("search_urls", [])
    scraped_text = ""

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    for url in urls:
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.extract()

                text = soup.get_text(separator=' ', strip=True)
                # Keep some reasonable length limit per URL
                scraped_text += f"\n\nSource: {url}\n{text[:5000]}"
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            continue

    if not scraped_text.strip():
        scraped_text = "No data could be scraped."

    return {"scraped_data": scraped_text}

def data_extraction_synthesis(state: AgentState) -> AgentState:
    """
    Pass the raw data through an LLM to extract the targeted business metrics,
    filter out noise, and generate a final Markdown report.
    """
    company_name = state["company_name"]
    scraped_data = state.get("scraped_data", "")
    openai_api_key = os.environ.get("OPENAI_API_KEY")

    if not openai_api_key:
        print("Warning: OPENAI_API_KEY not found. Skipping synthesis.")
        return {"final_report": "OPENAI_API_KEY not found."}

    system_prompt = (
        "You are an elite Corporate Intelligence Researcher. Your mission is to investigate a target company provided by the user, "
        "scrape relevant web data, and produce a highly structured, data-driven intelligence report.\n\n"
        "Your Execution Loop:\n"
        "Plan: Identify the search queries needed to find the company's official site, corporate profiles (like LinkedIn or Crunchbase), and recent financial press.\n"
        "Search & Scrape: Deploy your tools to extract raw text from these URLs. Look specifically for quantitative data.\n"
        "Synthesize: Cross-reference the data, discard marketing fluff, and extract hard facts.\n\n"
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

    user_prompt = f"Target Company: {company_name}\n\nScraped Data:\n{scraped_data[:15000]}"

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
        data = response.json()
        final_report = data["choices"][0]["message"]["content"]
        return {"final_report": final_report}
    except Exception as e:
        print(f"Error during data_extraction_synthesis: {e}")
        return {"final_report": f"Error generating report: {e}"}

# Graph Compilation
builder = StateGraph(AgentState)
builder.add_node("search_discovery", search_discovery)
builder.add_node("deep_scraper", deep_scraper)
builder.add_node("data_extraction_synthesis", data_extraction_synthesis)

builder.set_entry_point("search_discovery")
builder.add_edge("search_discovery", "deep_scraper")
builder.add_edge("deep_scraper", "data_extraction_synthesis")
builder.add_edge("data_extraction_synthesis", END)

graph = builder.compile()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        company = sys.argv[1]
        print(f"Starting analysis for: {company}")
        initial_state = {"company_name": company}
        final_state = graph.invoke(initial_state)
        print("\n=== FINAL REPORT ===\n")
        print(final_state.get("final_report", "No report generated."))
    else:
        print("Usage: python corporate_intelligence_agent.py \"[Company Name]\"")
