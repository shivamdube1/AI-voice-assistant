import os
import sys
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

def search_discovery(state: AgentState):
    company_name = state["company_name"]
    tavily_api_key = os.environ.get("TAVILY_API_KEY")

    if not tavily_api_key:
        print("Warning: TAVILY_API_KEY not found in environment variables.")
        # Dummy URLs for testing without API key
        return {"search_urls": [f"https://www.{company_name.lower().replace(' ', '')}.com", f"https://www.linkedin.com/company/{company_name.lower().replace(' ', '')}"]}

    query = f"{company_name} official website, LinkedIn, recent financial and business news"

    response = requests.post(
        "https://api.tavily.com/search",
        json={
            "api_key": tavily_api_key,
            "query": query,
            "search_depth": "advanced",
            "max_results": 5
        }
    )

    urls = []
    if response.status_code == 200:
        results = response.json().get("results", [])
        urls = [result["url"] for result in results]
    else:
        print(f"Error from Tavily API: {response.status_code} - {response.text}")

    return {"search_urls": urls}

def deep_scraper(state: AgentState):
    urls = state.get("search_urls", [])
    scraped_texts = []

    for url in urls:
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                # Extract text, removing script and style tags
                for script in soup(["script", "style"]):
                    script.extract()
                text = soup.get_text(separator=' ')
                # Clean up whitespace
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = '\\n'.join(chunk for chunk in chunks if chunk)

                # Limit text size to avoid blowing up context window
                scraped_texts.append(f"--- Data from {url} ---\\n{text[:5000]}")
            else:
                print(f"Failed to fetch {url} (Status: {response.status_code})")
        except Exception as e:
            print(f"Error scraping {url}: {e}")

    return {"scraped_data": "\\n\\n".join(scraped_texts)}

def data_extraction_synthesis(state: AgentState):
    company_name = state["company_name"]
    scraped_data = state.get("scraped_data", "")
    openai_api_key = os.environ.get("OPENAI_API_KEY")

    if not openai_api_key:
        print("Warning: OPENAI_API_KEY not found in environment variables.")
        return {"final_report": "# Dummy Report\\nNo API key provided."}

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

    user_prompt = f"Target Company: {company_name}\\n\\nScraped Data:\\n{scraped_data}\\n\\nPlease generate the intelligence report based strictly on the scraped data above."

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

    if response.status_code == 200:
        report = response.json()["choices"][0]["message"]["content"]
    else:
        print(f"Error from OpenAI API: {response.status_code} - {response.text}")
        report = f"Error generating report: {response.status_code}"

    return {"final_report": report}

def build_graph():
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("search_discovery", search_discovery)
    workflow.add_node("deep_scraper", deep_scraper)
    workflow.add_node("data_extraction_synthesis", data_extraction_synthesis)

    # Set entry point
    workflow.set_entry_point("search_discovery")

    # Add edges
    workflow.add_edge("search_discovery", "deep_scraper")
    workflow.add_edge("deep_scraper", "data_extraction_synthesis")
    workflow.add_edge("data_extraction_synthesis", END)

    return workflow.compile()

def main():
    if len(sys.argv) < 2:
        print('Usage: python corporate_intelligence_agent.py "[Company Name]"')
        sys.exit(1)

    company_name = sys.argv[1]

    print(f"Starting investigation on: {company_name}")

    graph = build_graph()

    initial_state = {
        "company_name": company_name,
        "search_urls": [],
        "scraped_data": "",
        "final_report": ""
    }

    try:
        # LangGraph invoke returns the final state
        result = graph.invoke(initial_state)
        print("\\n" + "="*50 + "\\n")
        print(result.get("final_report", "No report generated."))
        print("\\n" + "="*50 + "\\n")
    except Exception as e:
        print(f"Error executing graph: {e}")

if __name__ == "__main__":
    main()
