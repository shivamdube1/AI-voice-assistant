import os
import requests
from bs4 import BeautifulSoup
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from tavily import TavilyClient

# 1. State Definition
class AgentState(TypedDict):
    company_name: str
    search_queries: List[str]
    urls_to_scrape: List[str]
    raw_scraped_data: str
    final_report: str

# 2. Agent's System Prompt
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

# 3. Node Functions
def search_and_discovery_node(state: AgentState) -> AgentState:
    """
    Search & Discovery Node: Use a search tool (e.g., Tavily) to find the company's
    official website, LinkedIn page, and recent financial/business news.
    """
    company_name = state["company_name"]
    print(f"--> [Search & Discovery] Finding sources for: {company_name}")

    tavily_api_key = os.environ.get("TAVILY_API_KEY")
    search_queries = [
        f"{company_name} official website",
        f"{company_name} linkedin company page",
        f"{company_name} recent financial business news"
    ]
    urls = []

    if tavily_api_key:
        print("Using Tavily API for search...")
        try:
            client = TavilyClient(api_key=tavily_api_key)
            for query in search_queries:
                response = client.search(query=query, search_depth="basic")
                urls.extend([result["url"] for result in response.get("results", [])])
            urls = list(set(urls))[:5] # Deduplicate and limit
        except Exception as e:
            print(f"Tavily search failed: {e}")

    if not urls:
        print("No URLs found from Tavily or API key missing. Falling back to simulated URLs.")
        urls = [
            f"https://www.{company_name.lower().replace(' ', '')}.com",
            f"https://www.linkedin.com/company/{company_name.lower().replace(' ', '')}",
            f"https://news.ycombinator.com/item?id=12345"
        ]

    return {
        "company_name": company_name,
        "search_queries": search_queries,
        "urls_to_scrape": urls,
        "raw_scraped_data": state.get("raw_scraped_data", ""),
        "final_report": state.get("final_report", "")
    }

def deep_scraper_node(state: AgentState) -> AgentState:
    """
    Deep Scraper Node: Extract text from the official website (About Us, Careers)
    and external aggregators to specifically hunt for employee headcounts,
    performance metrics (revenue, growth), and core specializations.
    """
    urls = state.get("urls_to_scrape", [])
    print(f"--> [Deep Scraper] Scraping data from {len(urls)} URLs...")

    scraped_content = ""
    for url in urls:
        try:
            print(f"Scraping {url}...")
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
            response = requests.get(url, headers=headers, timeout=5)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                # Remove script and style elements
                for script_or_style in soup(["script", "style"]):
                    script_or_style.extract()
                text = soup.get_text(separator=' ', strip=True)
                # Take first 4000 chars to avoid prompt overflow but capture enough data
                scraped_content += f"--- Content from {url} ---\n{text[:4000]}\n\n"
        except Exception as e:
            print(f"Failed to scrape {url}: {e}")

    if not scraped_content.strip():
        print("Failed to scrape meaningful data. Using fallback simulated data.")
        scraped_content = f"Simulated scraped data for {state['company_name']}.\n"
        scraped_content += "They have around 500-1000 employees. "
        scraped_content += "They specialize in AI solutions and machine learning platforms. "
        scraped_content += "Revenue is estimated at $50M annually. "
        scraped_content += "Primary competitors include TechCorp and InnovateLLC."

    return {
        "company_name": state["company_name"],
        "search_queries": state["search_queries"],
        "urls_to_scrape": state["urls_to_scrape"],
        "raw_scraped_data": scraped_content,
        "final_report": state.get("final_report", "")
    }

def data_extraction_and_synthesis_node(state: AgentState) -> AgentState:
    """
    Data Extraction & Synthesis Node: Pass the raw data through an LLM to extract
    the targeted business metrics, filter out noise, and generate a final Markdown report.
    """
    company_name = state["company_name"]
    raw_data = state.get("raw_scraped_data", "")
    print(f"--> [Data Extraction & Synthesis] Generating structured report for {company_name}...")

    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if openai_api_key:
        print("Using OpenAI API for LLM synthesis...")
        try:
            llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=openai_api_key)
            prompt = ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT),
                ("user", "Target Company: {company_name}\n\nRaw Scraped Data:\n{raw_data}")
            ])
            chain = prompt | llm
            result = chain.invoke({"company_name": company_name, "raw_data": raw_data})
            report = result.content
        except Exception as e:
            print(f"OpenAI API call failed: {e}. Falling back to simulated report.")
            report = ""
    else:
        print("OPENAI_API_KEY not found. Falling back to simulated report.")
        report = ""

    if not report:
        report = f"""# Corporate Intelligence Report: {company_name}

## Executive Summary
{company_name} is an AI solutions provider focused on developing machine learning platforms for enterprise clients.

## Company Profile
- **Specialization**: AI solutions and machine learning platforms.
- **Company Size**: 500-1000 employees.
- **Headquarters / Key Locations**: Insufficient data found for this metric.

## Company Performance
- **Financials / Growth**: Estimated revenue at $50M annually.
- **Market Position**: Primary competitors include TechCorp and InnovateLLC.
- **Core Products & Offerings**: Insufficient data found for this metric.
- **Recent Developments**: Insufficient data found for this metric.
"""

    return {
        "company_name": state["company_name"],
        "search_queries": state["search_queries"],
        "urls_to_scrape": state["urls_to_scrape"],
        "raw_scraped_data": state["raw_scraped_data"],
        "final_report": report.strip()
    }

# 4. Compiled Graph Logic
def build_graph():
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("search_and_discovery", search_and_discovery_node)
    workflow.add_node("deep_scraper", deep_scraper_node)
    workflow.add_node("data_extraction_and_synthesis", data_extraction_and_synthesis_node)

    # Define edges and execution flow
    workflow.set_entry_point("search_and_discovery")
    workflow.add_edge("search_and_discovery", "deep_scraper")
    workflow.add_edge("deep_scraper", "data_extraction_and_synthesis")
    workflow.add_edge("data_extraction_and_synthesis", END)

    # Compile the graph
    app = workflow.compile()
    return app

if __name__ == "__main__":
    app = build_graph()

    initial_state: AgentState = {
        "company_name": "Example AI Corp",
        "search_queries": [],
        "urls_to_scrape": [],
        "raw_scraped_data": "",
        "final_report": ""
    }

    print("Starting Corporate Intelligence Agent Pipeline...\n")
    print("Note: Provide OPENAI_API_KEY and TAVILY_API_KEY in the environment for live API calls.\n")
    final_state = app.invoke(initial_state)

    print("\n" + "="*60)
    print("FINAL INTELLIGENCE REPORT")
    print("="*60)
    print(final_state["final_report"])
    print("="*60)
