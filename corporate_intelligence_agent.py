import sys
import os
import requests
from bs4 import BeautifulSoup
from typing import TypedDict, List
from langgraph.graph import StateGraph, END

# Define the state for the graph
class AgentState(TypedDict):
    company_name: str
    search_results: List[dict]
    scraped_data: str
    report: str

def search_discovery(state: AgentState) -> AgentState:
    """
    Search & Discovery Node: Use a search tool (e.g., Tavily) to find the company's official website,
    LinkedIn page, and recent financial/business news.
    """
    company_name = state["company_name"]
    tavily_api_key = os.environ.get("TAVILY_API_KEY")

    if not tavily_api_key:
        print("Warning: TAVILY_API_KEY environment variable not found. Mocking search results.")
        return {"search_results": [{"url": "mock_url_1", "content": "mock_content_1"}]}

    queries = [
        f"{company_name} official website",
        f"{company_name} linkedin company profile",
        f"{company_name} recent financial business news"
    ]

    all_results = []

    for query in queries:
        try:
            response = requests.post(
                "https://api.tavily.com/search",
                headers={"Content-Type": "application/json"},
                json={
                    "api_key": tavily_api_key,
                    "query": query,
                    "search_depth": "basic",
                    "include_answer": False,
                    "include_raw_content": False,
                    "max_results": 2
                }
            )
            response.raise_for_status()
            data = response.json()
            if "results" in data:
                all_results.extend(data["results"])
        except Exception as e:
            print(f"Error searching for {query}: {e}")

    # Remove duplicates
    seen_urls = set()
    unique_results = []
    for res in all_results:
        url = res.get("url")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_results.append(res)

    return {"search_results": unique_results}

def deep_scraper(state: AgentState) -> AgentState:
    """
    Deep Scraper Node: Extract text from the official website (About Us, Careers) and external aggregators
    to specifically hunt for employee headcounts, performance metrics (revenue, growth), and core specializations.
    """
    search_results = state.get("search_results", [])
    scraped_texts = []

    for result in search_results:
        url = result.get("url")
        if not url or url.startswith("mock_url"):
            continue

        try:
            # We add a timeout and user-agent to avoid being blocked easily
            headers = {"User-Agent": "Mozilla/5.0 (compatible; CorporateIntelligenceAgent/1.0)"}
            response = requests.get(url, headers=headers, timeout=5)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")

            # Extract text from paragraph and header tags
            tags_to_extract = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li'])
            page_text = " ".join([tag.get_text(strip=True) for tag in tags_to_extract])

            # Limit the text per page to avoid context window explosion
            if page_text:
                scraped_texts.append(f"Source: {url}\nContent: {page_text[:2000]}")

        except Exception as e:
            print(f"Error scraping {url}: {e}")

    combined_data = "\n\n".join(scraped_texts)

    # If no data was scraped (e.g. mocked or all failed), provide a fallback
    if not combined_data:
        combined_data = "No data could be scraped from the provided sources."

    return {"scraped_data": combined_data}

def data_extraction_synthesis(state: AgentState) -> AgentState:
    """
    Data Extraction & Synthesis Node: Pass the raw data through an LLM to extract the targeted
    business metrics, filter out noise, and generate a final Markdown report.
    """
    company_name = state["company_name"]
    scraped_data = state.get("scraped_data", "")
    openai_api_key = os.environ.get("OPENAI_API_KEY")

    system_prompt = """You are an elite Corporate Intelligence Researcher. Your mission is to investigate a target company provided by the user, scrape relevant web data, and produce a highly structured, data-driven intelligence report.

Required Report Structure:

# Executive Summary
A concise, one-paragraph overview of the company.

# Company Profile

* **Specialization**: What is their exact niche, core technology, or primary service?
* **Company Size**: Number of employees (provide an exact number or estimated range based on scraped data).
* **Headquarters / Key Locations**: Primary operational bases.

# Company Performance

* **Financials / Growth**: Estimated revenue, funding rounds, market share, or notable growth metrics.
* **Market Position**: Who are their primary competitors?
* **Core Products & Offerings**: The specific products or services they sell and their target demographic.
* **Recent Developments**: Key news, leadership changes, or major events from the last 6-12 months.

Constraints: Ground your entire report strictly in the data you scrape. If a specific metric (like revenue or exact employee count) cannot be found, explicitly state 'Insufficient data found for this metric' rather than guessing."""

    user_prompt = f"""Target Company: {company_name}

Here is the raw scraped data from various sources:
{scraped_data}

Please generate the structured Markdown report based ONLY on the provided data above. Remember to output 'Insufficient data found for this metric' if a piece of information is missing."""

    if not openai_api_key:
        print("Warning: OPENAI_API_KEY environment variable not found. Returning a dummy report.")
        dummy_report = f"# Executive Summary\n{company_name} is a mocked company.\n\n# Company Profile\n\n* **Specialization**: Insufficient data found for this metric\n* **Company Size**: Insufficient data found for this metric\n* **Headquarters / Key Locations**: Insufficient data found for this metric\n\n# Company Performance\n\n* **Financials / Growth**: Insufficient data found for this metric\n* **Market Position**: Insufficient data found for this metric\n* **Core Products & Offerings**: Insufficient data found for this metric\n* **Recent Developments**: Insufficient data found for this metric"
        return {"report": dummy_report}

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {openai_api_key}"
            },
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.2
            }
        )
        response.raise_for_status()
        data = response.json()
        report = data["choices"][0]["message"]["content"]
    except Exception as e:
        report = f"Error generating report with OpenAI: {e}"

    return {"report": report}

def build_graph():
    # Define a new graph
    workflow = StateGraph(AgentState)

    # Define the nodes we will cycle between
    workflow.add_node("search", search_discovery)
    workflow.add_node("scrape", deep_scraper)
    workflow.add_node("synthesize", data_extraction_synthesis)

    # Set the entrypoint
    workflow.set_entry_point("search")

    # Add edges
    workflow.add_edge("search", "scrape")
    workflow.add_edge("scrape", "synthesize")
    workflow.add_edge("synthesize", END)

    # Compile
    app = workflow.compile()
    return app

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python corporate_intelligence_agent.py \"[Company Name]\"")
        sys.exit(1)

    company_name = sys.argv[1]

    app = build_graph()

    initial_state = {"company_name": company_name}

    print(f"Starting investigation for: {company_name}...")

    final_state = app.invoke(initial_state)

    print("\n" + "="*50 + "\n")
    print(final_state.get("report", "No report generated."))
    print("\n" + "="*50 + "\n")
