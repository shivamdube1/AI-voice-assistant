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

def search_discovery(state: AgentState) -> dict:
    """
    Search & Discovery Node:
    Uses Tavily API directly via requests to find the company's official website,
    LinkedIn page, and recent financial/business news.
    """
    company_name = state["company_name"]
    print(f"[*] Running Search & Discovery for: {company_name}")

    tavily_api_key = os.environ.get("TAVILY_API_KEY")
    if not tavily_api_key:
        print("[!] TAVILY_API_KEY not found. Operating with empty URLs.")
        return {"search_urls": []}

    query = f"{company_name} official website, LinkedIn, financial business news"

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
        response = requests.post("https://api.tavily.com/search", json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()

        urls = []
        if "results" in data:
            urls = [result["url"] for result in data["results"] if "url" in result]

        print(f"[*] Found {len(urls)} URLs for {company_name}")
        return {"search_urls": urls}

    except Exception as e:
        print(f"[!] Error during Tavily search: {e}")
        return {"search_urls": []}

def deep_scraper(state: AgentState) -> dict:
    """
    Deep Scraper Node:
    Extract text from the official website (About Us, Careers) and external aggregators
    to hunt for employee headcounts, performance metrics, and core specializations.
    """
    print(f"[*] Running Deep Scraper for {len(state['search_urls'])} URLs...")
    all_text = ""

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    for url in state["search_urls"]:
        try:
            print(f"  -> Scraping: {url}")
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')

                # Strip out script and style tags
                for script_or_style in soup(['script', 'style']):
                    script_or_style.extract()

                text = soup.get_text(separator=' ')
                # Clean up whitespace
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = ' '.join(chunk for chunk in chunks if chunk)

                # Add to all_text (truncate to avoid massive payloads)
                all_text += f"\n\n--- Content from {url} ---\n"
                all_text += text[:5000] # Cap per URL to prevent token limits

        except Exception as e:
            print(f"  [!] Failed to scrape {url}: {e}")

    print("[*] Scraping completed.")
    return {"scraped_data": all_text}


def data_extraction_synthesis(state: AgentState) -> dict:
    """
    Data Extraction & Synthesis Node:
    Pass the raw data through an LLM to extract targeted business metrics, filter out noise,
    and generate a final Markdown report.
    """
    print("[*] Running Data Extraction & Synthesis...")

    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        print("[!] OPENAI_API_KEY not found. Returning a placeholder report.")
        return {"final_report": "Error: OPENAI_API_KEY not set. Cannot synthesize report."}

    company_name = state["company_name"]
    scraped_data = state["scraped_data"]

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

    user_prompt = f"Target Company: {company_name}\n\nHere is the scraped raw data from the search URLs:\n\n{scraped_data}"

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
        response = requests.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        report = data["choices"][0]["message"]["content"]
        print("[*] Report synthesis complete.")
        return {"final_report": report}
    except Exception as e:
        print(f"[!] Error during OpenAI synthesis: {e}")
        return {"final_report": f"Error synthesizing report: {e}"}

def build_graph():
    """Assemble the LangGraph pipeline."""
    graph_builder = StateGraph(AgentState)

    # Add Nodes
    graph_builder.add_node("search_discovery", search_discovery)
    graph_builder.add_node("deep_scraper", deep_scraper)
    graph_builder.add_node("data_extraction_synthesis", data_extraction_synthesis)

    # Add Edges (linear flow)
    graph_builder.set_entry_point("search_discovery")
    graph_builder.add_edge("search_discovery", "deep_scraper")
    graph_builder.add_edge("deep_scraper", "data_extraction_synthesis")
    graph_builder.add_edge("data_extraction_synthesis", END)

    return graph_builder.compile()

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python corporate_intelligence_agent.py \"[Company Name]\"")
        sys.exit(1)

    target_company = sys.argv[1]

    # Run the graph
    app = build_graph()

    initial_state = {
        "company_name": target_company,
        "search_urls": [],
        "scraped_data": "",
        "final_report": ""
    }

    print(f"==================================================")
    print(f"Starting Corporate Intelligence Agent for: {target_company}")
    print(f"==================================================\n")

    final_state = app.invoke(initial_state)

    print(f"\n==================================================")
    print(f"FINAL INTELLIGENCE REPORT")
    print(f"==================================================\n")
    print(final_state["final_report"])
    print(f"\n==================================================")
