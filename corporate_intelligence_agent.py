import os
import requests
from bs4 import BeautifulSoup
from typing import Dict, TypedDict, List, Any
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

# Define the state for the LangGraph
class AgentState(TypedDict):
    company_name: str
    search_results: List[Dict[str, Any]]
    scraped_data: str
    report: str

def search_and_discovery(state: AgentState) -> AgentState:
    """
    Search & Discovery Node: Use a search tool (e.g., Tavily) to find the company's
    official website, LinkedIn page, and recent financial/business news.
    """
    company_name = state["company_name"]

    # Initialize Tavily Search Tool
    # Note: Requires TAVILY_API_KEY to be set in environment
    tavily_tool = TavilySearchResults(max_results=5)

    queries = [
        f"{company_name} official website",
        f"{company_name} LinkedIn page",
        f"{company_name} recent financial business news"
    ]

    all_results = []
    for query in queries:
        try:
            results = tavily_tool.invoke({"query": query})
            if isinstance(results, list):
                all_results.extend(results)
        except Exception as e:
            print(f"Error searching for '{query}': {e}")

    # Deduplicate results based on URL
    unique_urls = set()
    deduped_results = []
    for result in all_results:
        if isinstance(result, dict) and "url" in result:
            if result["url"] not in unique_urls:
                unique_urls.add(result["url"])
                deduped_results.append(result)

    return {"search_results": deduped_results}

def deep_scraper(state: AgentState) -> AgentState:
    """
    Deep Scraper Node: Extract text from the official website (About Us, Careers)
    and external aggregators to specifically hunt for employee headcounts,
    performance metrics (revenue, growth), and core specializations.
    """
    search_results = state.get("search_results", [])
    scraped_text_parts = []

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    for result in search_results:
        url = result.get("url")
        if not url:
            continue

        content = result.get("content", "")
        scraped_text_parts.append(f"Source: {url}\nSummary from Search: {content}\n")

        # Attempt direct scrape for deeper content
        try:
            response = requests.get(url, headers=headers, timeout=5)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')

                # Remove scripts and styles
                for script in soup(["script", "style"]):
                    script.extract()

                text = soup.get_text(separator=' ', strip=True)

                # Limit text length to avoid context window explosion (e.g., max 2000 chars per site)
                snippet = text[:2000] if len(text) > 2000 else text
                scraped_text_parts.append(f"Extracted Page Text (Snippet): {snippet}...\n")
        except Exception as e:
            # Ignore scrape errors (e.g., timeouts, anti-bot protections)
            pass

    combined_scraped_data = "\n".join(scraped_text_parts)
    return {"scraped_data": combined_scraped_data}

def data_extraction_and_synthesis(state: AgentState) -> AgentState:
    """
    Data Extraction & Synthesis Node: Pass the raw data through an LLM to extract
    the targeted business metrics, filter out noise, and generate a final Markdown report.
    """
    company_name = state["company_name"]
    scraped_data = state.get("scraped_data", "")

    # Initialize LLM
    # Note: Requires OPENAI_API_KEY to be set in environment
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

    human_message = f"Target Company: {company_name}\n\nRaw Scraped Data:\n{scraped_data}\n\nPlease generate the required structured intelligence report based on the raw data."

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_message)
    ]

    response = llm.invoke(messages)

    return {"report": response.content}

# Compile the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("search_and_discovery", search_and_discovery)
workflow.add_node("deep_scraper", deep_scraper)
workflow.add_node("data_extraction_and_synthesis", data_extraction_and_synthesis)

# Set entry point
workflow.set_entry_point("search_and_discovery")

# Add edges
workflow.add_edge("search_and_discovery", "deep_scraper")
workflow.add_edge("deep_scraper", "data_extraction_and_synthesis")
workflow.add_edge("data_extraction_and_synthesis", END)

# Compile
app = workflow.compile()

def run_agent(company_name: str):
    """
    Helper function to run the compiled graph logic.
    """
    initial_state = {"company_name": company_name}

    print(f"Starting intelligence gathering for: {company_name}...\n")

    try:
        # Stream the graph execution
        final_state = None
        for s in app.stream(initial_state):
            if "search_and_discovery" in s:
                print("Completed Search & Discovery phase.")
            elif "deep_scraper" in s:
                print("Completed Deep Scraper phase.")
            elif "data_extraction_and_synthesis" in s:
                print("Completed Data Extraction & Synthesis phase.")
                final_state = s["data_extraction_and_synthesis"]

        # Get the final report state
        report = final_state.get("report", "") if final_state else ""

        print("\n" + "="*50)
        print("FINAL INTELLIGENCE REPORT")
        print("="*50 + "\n")
        print(report)
        return report

    except Exception as e:
        print(f"An error occurred during execution: {e}")
        return None

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        company_to_research = " ".join(sys.argv[1:])
    else:
        company_to_research = "Anthropic" # Default for testing

    run_agent(company_to_research)
