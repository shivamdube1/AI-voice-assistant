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

def search_discovery(state: AgentState) -> AgentState:
    """Use Tavily API to find the company's official website, LinkedIn page, and recent news."""
    company_name = state["company_name"]
    tavily_api_key = os.environ.get("TAVILY_API_KEY", "")

    if not tavily_api_key:
        print("Warning: TAVILY_API_KEY not found in environment. Using mock URLs.")
        state["search_urls"] = [
            f"https://www.{company_name.lower().replace(' ', '')}.com",
            f"https://www.linkedin.com/company/{company_name.lower().replace(' ', '')}"
        ]
        return state

    query = f"{company_name} official website, LinkedIn, recent business financial news"

    try:
        response = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": tavily_api_key,
                "query": query,
                "search_depth": "basic",
                "max_results": 5
            },
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        urls = [result["url"] for result in data.get("results", [])]
        state["search_urls"] = urls
    except Exception as e:
        print(f"Error calling Tavily API: {e}")
        state["search_urls"] = []

    return state

def deep_scraper(state: AgentState) -> AgentState:
    """Extract text from the identified URLs specifically hunting for company metrics."""
    urls = state.get("search_urls", [])
    scraped_texts = []

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    for url in urls:
        try:
            # We skip linkedin.com as it generally blocks basic requests
            if "linkedin.com" in url:
                continue

            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            # Remove scripts and styles
            for script in soup(["script", "style"]):
                script.decompose()

            text = soup.get_text(separator=' ', strip=True)
            # Take only a chunk of text to prevent context limits
            scraped_texts.append(f"Source ({url}):\n{text[:3000]}")
        except Exception as e:
            print(f"Error scraping {url}: {e}")

    state["scraped_data"] = "\n\n".join(scraped_texts)

    # If no real data was scraped (e.g. testing), provide mock data
    if not state["scraped_data"]:
        state["scraped_data"] = f"Mock data for {state['company_name']}: Estimated 500 employees, $50M revenue, AI industry, based in SF."

    return state

def data_extraction_synthesis(state: AgentState) -> AgentState:
    """Pass raw data through an LLM to extract metrics and generate a Markdown report."""
    company_name = state["company_name"]
    scraped_data = state.get("scraped_data", "")
    openai_api_key = os.environ.get("OPENAI_API_KEY", "")

    system_prompt = (
        "You are an elite Corporate Intelligence Researcher. Your mission is to investigate "
        "a target company provided by the user, scrape relevant web data, and produce a highly structured, data-driven intelligence report.\n\n"
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
        "Constraints: Ground your entire report strictly in the data you scrape. If a specific metric "
        "(like revenue or exact employee count) cannot be found, explicitly state 'Insufficient data found for this metric' rather than guessing."
    )

    user_prompt = f"Target Company: {company_name}\n\nScraped Data:\n{scraped_data}\n\nPlease generate the required structured report based strictly on the data above."

    if not openai_api_key:
        print("Warning: OPENAI_API_KEY not found in environment. Using mock LLM response.")
        state["final_report"] = f"# Corporate Intelligence Report: {company_name}\n\n" \
                                "## Executive Summary\n" \
                                f"An overview of {company_name} based on provided data.\n\n" \
                                "## Company Profile\n" \
                                f"- Specialization: Industry details for {company_name}.\n" \
                                "- Company Size: Estimated 500 employees.\n" \
                                "- Headquarters / Key Locations: SF.\n\n" \
                                "## Company Performance\n" \
                                "- Financials / Growth: $50M revenue.\n" \
                                "- Market Position: Insufficient data found for this metric.\n" \
                                "- Core Products & Offerings: Insufficient data found for this metric.\n" \
                                "- Recent Developments: Insufficient data found for this metric."
        return state

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {openai_api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.2
            },
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        state["final_report"] = data["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        state["final_report"] = f"Failed to generate report due to error: {e}"

    return state

# Graph Compilation
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
        print("Usage: python corporate_intelligence_agent.py \"[Company Name]\"")
        sys.exit(1)

    company_name = sys.argv[1]
    print(f"Starting investigation for: {company_name}...\n")

    initial_state = {
        "company_name": company_name,
        "search_urls": [],
        "scraped_data": "",
        "final_report": ""
    }

    # Run the graph
    result = app.invoke(initial_state)

    print("\n" + "="*50 + "\n")
    print(result.get("final_report", "No report generated."))
    print("\n" + "="*50 + "\n")
