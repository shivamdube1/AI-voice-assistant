import os
import sys
import requests
from typing import TypedDict, List
from bs4 import BeautifulSoup
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END

# Define State
class AgentState(TypedDict):
    company_name: str
    urls_to_scrape: List[str]
    raw_scraped_data: str
    report: str

# Search & Discovery Node
def search_discovery_node(state: AgentState):
    company_name = state["company_name"]
    print(f"--- SEARCH & DISCOVERY: {company_name} ---")

    # Initialize Tavily search tool
    tavily_tool = TavilySearchResults(max_results=5)

    queries = [
        f"{company_name} official website",
        f"{company_name} LinkedIn",
        f"{company_name} recent financial business news"
    ]

    urls = []
    for query in queries:
        try:
            results = tavily_tool.invoke({"query": query})
            for res in results:
                if 'url' in res:
                    urls.append(res['url'])
        except Exception as e:
            print(f"Error searching for {query}: {e}")

    # Deduplicate URLs while preserving order
    unique_urls = []
    for url in urls:
        if url not in unique_urls:
            unique_urls.append(url)

    return {"urls_to_scrape": unique_urls[:10]} # Limit to 10 URLs to process

# Deep Scraper Node
def deep_scraper_node(state: AgentState):
    urls = state.get("urls_to_scrape", [])
    print(f"--- DEEP SCRAPER: Scraping {len(urls)} URLs ---")

    scraped_texts = []
    for url in urls:
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, headers=headers, timeout=5)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                text = soup.get_text(separator=' ', strip=True)
                scraped_texts.append(f"Source: {url}\nContent: {text[:2500]}") # Keep it reasonably sized
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            continue

    combined_data = "\n\n---\n\n".join(scraped_texts)
    return {"raw_scraped_data": combined_data}

# Data Extraction & Synthesis Node
def data_extraction_synthesis_node(state: AgentState):
    company_name = state["company_name"]
    raw_data = state.get("raw_scraped_data", "")
    print(f"--- DATA EXTRACTION & SYNTHESIS: {company_name} ---")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

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

    user_prompt = f"Target Company: {company_name}\n\nHere is the raw scraped data:\n{raw_data[:25000]}"

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "{user_prompt}")
    ])

    chain = prompt | llm
    response = chain.invoke({"user_prompt": user_prompt})

    return {"report": response.content}

# Compile Graph Logic
def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("search_discovery", search_discovery_node)
    workflow.add_node("deep_scraper", deep_scraper_node)
    workflow.add_node("data_extraction_synthesis", data_extraction_synthesis_node)

    workflow.set_entry_point("search_discovery")
    workflow.add_edge("search_discovery", "deep_scraper")
    workflow.add_edge("deep_scraper", "data_extraction_synthesis")
    workflow.add_edge("data_extraction_synthesis", END)

    return workflow.compile()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python corporate_intelligence_agent.py \"[Company Name]\"")
        sys.exit(1)

    company_name = sys.argv[1]

    if "OPENAI_API_KEY" not in os.environ or "TAVILY_API_KEY" not in os.environ:
        print("Warning: OPENAI_API_KEY and TAVILY_API_KEY environment variables should be set for live API calls.")

    app = build_graph()
    initial_state = {"company_name": company_name}

    print(f"Starting Corporate Intelligence Agent for '{company_name}'...\n")
    final_state = app.invoke(initial_state)

    print("\n" + "="*50)
    print("FINAL INTELLIGENCE REPORT:")
    print("="*50 + "\n")
    print(final_state.get("report", "No report generated."))
