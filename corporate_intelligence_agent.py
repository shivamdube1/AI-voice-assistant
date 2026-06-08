import os
import json
import requests
from bs4 import BeautifulSoup
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, START, END

# Define the state for LangGraph
class AgentState(TypedDict):
    company_name: str
    search_queries: List[str]
    scraped_urls: List[str]
    raw_scraped_data: str
    report: str

def get_openai_chat_completion(messages: List[Dict[str, str]], model: str = "gpt-4o", temperature: float = 0.2) -> str:
    """Helper to make direct calls to OpenAI API using requests."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is missing.")

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature
    }
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

def search_tavily(query: str, max_results: int = 3) -> List[Dict[str, Any]]:
    """Helper to make direct calls to Tavily API using requests."""
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY environment variable is missing.")

    url = "https://api.tavily.com/search"
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "api_key": api_key,
        "query": query,
        "max_results": max_results,
        "search_depth": "basic",
        "include_answer": False
    }
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json().get("results", [])

def search_discovery(state: AgentState) -> AgentState:
    """Node 1: Formulate queries and collect URLs."""
    company_name = state["company_name"]

    # 1. Ask LLM to generate targeted search queries
    prompt = f"""You are a research assistant. Provide exactly 3 search queries to investigate the company "{company_name}".
1. A query to find their official website.
2. A query to find their corporate profile (like LinkedIn or Crunchbase).
3. A query to find recent financial/business news.
Return ONLY the 3 queries as a JSON list of strings, with no markdown formatting or extra text."""

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

    try:
        response_text = get_openai_chat_completion(messages)
        # Clean up markdown if any
        if response_text.startswith("```json"):
            response_text = response_text[7:-3]
        elif response_text.startswith("```"):
            response_text = response_text[3:-3]
        queries = json.loads(response_text.strip())
    except Exception as e:
        # Fallback queries
        queries = [
            f"{company_name} official website",
            f"{company_name} LinkedIn profile",
            f"{company_name} recent financial business news"
        ]

    # 2. Search using Tavily
    scraped_urls = []
    for query in queries:
        try:
            results = search_tavily(query, max_results=2)
            for res in results:
                url = res.get("url")
                if url and url not in scraped_urls:
                    scraped_urls.append(url)
        except Exception as e:
            print(f"Search failed for query '{query}': {e}")

    return {"search_queries": queries, "scraped_urls": scraped_urls}

def deep_scraper(state: AgentState) -> AgentState:
    """Node 2: Extract text from URLs."""
    urls = state["scraped_urls"]
    raw_data = ""

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    for url in urls:
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                # Remove script and style elements
                for script_or_style in soup(["script", "style", "nav", "footer", "header"]):
                    script_or_style.extract()

                text = soup.get_text(separator=' ', strip=True)
                # Keep it reasonably short to fit context windows
                raw_data += f"\n--- Data from {url} ---\n"
                raw_data += text[:5000] + "\n"
        except Exception as e:
            raw_data += f"\n--- Failed to scrape {url}: {e} ---\n"

    return {"raw_scraped_data": raw_data}

def data_extraction_synthesis(state: AgentState) -> AgentState:
    """Node 3: Generate the report using LLM."""
    company_name = state["company_name"]
    raw_data = state["raw_scraped_data"]

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

Constraints: Ground your entire report strictly in the data you scrape. If a specific metric (like revenue or exact employee count) cannot be found, explicitly state 'Insufficient data found for this metric' rather than guessing."""

    user_prompt = f"""Target Company: {company_name}

Here is the raw scraped data from various sources:
{raw_data}

Please generate the structured Markdown report based on this data, following the exact required structure and constraints."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    report = get_openai_chat_completion(messages)
    return {"report": report}

def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("search_discovery", search_discovery)
    workflow.add_node("deep_scraper", deep_scraper)
    workflow.add_node("data_extraction_synthesis", data_extraction_synthesis)

    workflow.add_edge(START, "search_discovery")
    workflow.add_edge("search_discovery", "deep_scraper")
    workflow.add_edge("deep_scraper", "data_extraction_synthesis")
    workflow.add_edge("data_extraction_synthesis", END)

    return workflow.compile()

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print('Usage: python corporate_intelligence_agent.py "[Company Name]"')
        sys.exit(1)

    company = sys.argv[1]
    print(f"Starting research for: {company}")

    app = build_graph()

    # Run the graph
    initial_state = {"company_name": company, "search_queries": [], "scraped_urls": [], "raw_scraped_data": "", "report": ""}

    try:
        final_state = app.invoke(initial_state)
        print("\n" + "="*50 + "\n")
        print(final_state["report"])
        print("\n" + "="*50 + "\n")
    except Exception as e:
        print(f"An error occurred: {e}")
