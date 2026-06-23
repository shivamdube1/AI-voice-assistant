import os
import sys
import json
import requests
from bs4 import BeautifulSoup
from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END

class AgentState(TypedDict):
    company_name: str
    urls: List[str]
    scraped_texts: List[str]
    report: str

def search_discovery(state: AgentState) -> dict:
    company_name = state["company_name"]
    tavily_api_key = os.environ.get("TAVILY_API_KEY")
    if not tavily_api_key:
        print("Warning: TAVILY_API_KEY is not set.")

    query = f"{company_name} official website OR LinkedIn OR recent financial news OR revenue OR employee count"
    url = "https://api.tavily.com/search"
    headers = {"Content-Type": "application/json"}
    payload = {
        "api_key": tavily_api_key,
        "query": query,
        "search_depth": "advanced",
        "include_answer": False,
        "include_images": False,
        "include_raw_content": False,
        "max_results": 5
    }

    urls = []
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        urls = [result.get("url") for result in data.get("results", []) if result.get("url")]
    except Exception as e:
        print(f"Error during Tavily search: {e}")

    return {"urls": urls}

def deep_scraper(state: AgentState) -> dict:
    urls = state.get("urls", [])
    scraped_texts = []

    for url in urls:
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            # Remove scripts, styles, etc.
            for element in soup(["script", "style", "nav", "footer", "header"]):
                element.extract()

            text = soup.get_text(separator=' ')
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)

            # Limit length to avoid blowing up context window
            scraped_texts.append(text[:2000])
        except Exception as e:
            print(f"Error scraping {url}: {e}")

    return {"scraped_texts": scraped_texts}


def data_extraction_synthesis(state: AgentState) -> dict:
    company_name = state["company_name"]
    scraped_texts = state.get("scraped_texts", [])
    openai_api_key = os.environ.get("OPENAI_API_KEY")

    if not openai_api_key:
        print("Warning: OPENAI_API_KEY is not set.")

    combined_text = "\n\n".join(scraped_texts)

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

Constraints: Ground your entire report strictly in the data you scrape. If a specific metric (like revenue or exact employee count) cannot be found, explicitly state 'Insufficient data found for this metric' rather than guessing. Format the output in Markdown."""

    user_prompt = f"Target Company: {company_name}\n\nScraped Data:\n{combined_text}\n\nPlease generate the report based ONLY on the scraped data above."

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
        "temperature": 0
    }

    report = ""
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        report = data["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error during OpenAI synthesis: {e}")
        report = "Error generating report."

    return {"report": report}


# LangGraph Setup
workflow = StateGraph(AgentState)

workflow.add_node("search_discovery", search_discovery)
workflow.add_node("deep_scraper", deep_scraper)
workflow.add_node("data_extraction_synthesis", data_extraction_synthesis)

workflow.add_edge(START, "search_discovery")
workflow.add_edge("search_discovery", "deep_scraper")
workflow.add_edge("deep_scraper", "data_extraction_synthesis")
workflow.add_edge("data_extraction_synthesis", END)

app = workflow.compile()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python corporate_intelligence_agent.py \"[Company Name]\"")
        sys.exit(1)

    company_name = sys.argv[1]
    initial_state = {"company_name": company_name}

    print(f"Starting investigation for: {company_name}...\n")
    final_state = app.invoke(initial_state)

    print("\n" + "="*50 + "\n")
    print(final_state.get("report", "No report generated."))
    print("\n" + "="*50 + "\n")
