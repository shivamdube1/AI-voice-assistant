import os
import sys
import requests
from bs4 import BeautifulSoup
from typing import TypedDict, List, Dict, Any
from tavily import TavilyClient
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END

# Define the LangGraph State
class AgentState(TypedDict):
    company_name: str
    search_queries: List[str]
    search_results: List[Dict[str, Any]]
    scraped_data: List[str]
    report: str

SYSTEM_PROMPT = """You are an elite Corporate Intelligence Researcher. Your mission is to investigate a target company provided by the user, scrape relevant web data, and produce a highly structured, data-driven intelligence report.

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

def search_discovery(state: AgentState) -> AgentState:
    """Uses Tavily to find the company's official website, LinkedIn page, and recent news."""
    company_name = state["company_name"]
    tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))

    print(f"[*] Search & Discovery: Researching {company_name}...")

    queries = [
        f"{company_name} official website",
        f"{company_name} company linkedin profile",
        f"{company_name} recent financial business news"
    ]

    search_results = []
    for query in queries:
        try:
            response = tavily_client.search(query=query, search_depth="advanced", max_results=3)
            search_results.extend(response.get("results", []))
        except Exception as e:
            print(f"Error searching for '{query}': {e}")

    return {"search_results": search_results}

def deep_scraper(state: AgentState) -> AgentState:
    """Extracts text from URLs to hunt for headcount, metrics, and specializations."""
    search_results = state.get("search_results", [])
    scraped_data = []

    print(f"[*] Deep Scraper: Scraping {len(search_results)} URLs...")

    # We only want to scrape a limited number of unique URLs to save time and context length
    urls_to_scrape = list(set([result.get("url") for result in search_results if result.get("url")]))[:5]

    for url in urls_to_scrape:
        try:
            # Setting a timeout to prevent hanging
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                # Extract text, removing script and style elements
                for script_or_style in soup(['script', 'style']):
                    script_or_style.extract()
                text = soup.get_text(separator=' ', strip=True)
                # Keep a manageable chunk of text per URL
                scraped_data.append(f"Source: {url}\nContent:\n{text[:2000]}")
        except Exception as e:
            print(f"Error scraping {url}: {e}")

    return {"scraped_data": scraped_data}

def data_extraction_synthesis(state: AgentState) -> AgentState:
    """Passes raw data through LLM to extract metrics and generate final report."""
    company_name = state["company_name"]
    scraped_data = state.get("scraped_data", [])

    print(f"[*] Data Extraction & Synthesis: Generating report for {company_name}...")

    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # Combine scraped data into one string
    combined_data = "\n\n".join(scraped_data)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Company to investigate: {company_name}\n\nHere is the raw scraped data from the web:\n{combined_data}\n\nPlease generate the required structured Markdown report based STRICTLY on this data.")
    ]

    response = llm.invoke(messages)

    return {"report": response.content}

# Build the LangGraph
def build_graph():
    # Initialize the graph with the AgentState
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("search_discovery", search_discovery)
    workflow.add_node("deep_scraper", deep_scraper)
    workflow.add_node("data_extraction_synthesis", data_extraction_synthesis)

    # Define edges
    workflow.set_entry_point("search_discovery")
    workflow.add_edge("search_discovery", "deep_scraper")
    workflow.add_edge("deep_scraper", "data_extraction_synthesis")
    workflow.add_edge("data_extraction_synthesis", END)

    # Compile the graph
    app = workflow.compile()
    return app

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python corporate_intelligence_agent.py \"[Company Name]\"")
        sys.exit(1)

    target_company = sys.argv[1]

    # Check for required API keys
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is missing.")
        sys.exit(1)

    if not os.environ.get("TAVILY_API_KEY"):
        print("Error: TAVILY_API_KEY environment variable is missing.")
        sys.exit(1)

    # Build the graph
    app = build_graph()

    # Initialize the state
    initial_state = {
        "company_name": target_company,
        "search_queries": [],
        "search_results": [],
        "scraped_data": [],
        "report": ""
    }

    print(f"\n=======================================================")
    print(f"Starting Corporate Intelligence Agent for: {target_company}")
    print(f"=======================================================\n")

    # Run the graph
    try:
        final_state = app.invoke(initial_state)

        print("\n=======================================================")
        print(f"Final Report for {target_company}")
        print(f"=======================================================\n")
        print(final_state["report"])

    except Exception as e:
        print(f"\nAn error occurred during execution: {e}")
