import unittest
from unittest.mock import patch, MagicMock
import os
import sys

# Import the modules from the agent script
from corporate_intelligence_agent import (
    search_discovery,
    deep_scraper,
    data_extraction_synthesis,
    build_graph,
    AgentState
)

class TestCorporateIntelligenceAgent(unittest.TestCase):
    def setUp(self):
        # Setup environment variables for testing
        os.environ["TAVILY_API_KEY"] = "fake_tavily_key"
        os.environ["OPENAI_API_KEY"] = "fake_openai_key"

        self.initial_state = {
            "company_name": "TestCorp",
            "search_urls": [],
            "scraped_data": "",
            "final_report": ""
        }

    @patch('corporate_intelligence_agent.requests.post')
    def test_search_discovery(self, mock_post):
        # Mock Tavily response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {"url": "https://testcorp.com"},
                {"url": "https://linkedin.com/company/testcorp"}
            ]
        }
        mock_post.return_value = mock_response

        state_update = search_discovery(self.initial_state)
        self.assertIn("search_urls", state_update)
        self.assertEqual(len(state_update["search_urls"]), 2)
        self.assertEqual(state_update["search_urls"][0], "https://testcorp.com")

    @patch('corporate_intelligence_agent.requests.get')
    def test_deep_scraper(self, mock_get):
        # Mock website response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"<html><body><h1>TestCorp</h1><p>We are a test company with 100 employees.</p></body></html>"
        mock_get.return_value = mock_response

        state_with_urls = self.initial_state.copy()
        state_with_urls["search_urls"] = ["https://testcorp.com"]

        state_update = deep_scraper(state_with_urls)

        self.assertIn("scraped_data", state_update)
        self.assertIn("TestCorp", state_update["scraped_data"])
        self.assertIn("100 employees", state_update["scraped_data"])

    @patch('corporate_intelligence_agent.requests.post')
    def test_data_extraction_synthesis(self, mock_post):
        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "# TestCorp Intelligence Report\n\nExecutive Summary..."
                    }
                }
            ]
        }
        mock_post.return_value = mock_response

        state_with_data = self.initial_state.copy()
        state_with_data["scraped_data"] = "Raw text about TestCorp."

        state_update = data_extraction_synthesis(state_with_data)

        self.assertIn("final_report", state_update)
        self.assertIn("TestCorp Intelligence Report", state_update["final_report"])

    @patch('corporate_intelligence_agent.requests.post')
    @patch('corporate_intelligence_agent.requests.get')
    def test_full_graph_execution(self, mock_get, mock_post):
        # Mock Tavily response (first post call)
        mock_tavily_response = MagicMock()
        mock_tavily_response.status_code = 200
        mock_tavily_response.json.return_value = {
            "results": [{"url": "https://testcorp.com"}]
        }

        # Mock OpenAI response (second post call)
        mock_openai_response = MagicMock()
        mock_openai_response.status_code = 200
        mock_openai_response.json.return_value = {
            "choices": [{"message": {"content": "Final Markdown Report"}}]
        }

        # Set side effect for post to differentiate Tavily vs OpenAI based on URL
        def post_side_effect(url, *args, **kwargs):
            if "tavily" in url:
                return mock_tavily_response
            elif "openai" in url:
                return mock_openai_response
            return MagicMock()

        mock_post.side_effect = post_side_effect

        # Mock Scraper response
        mock_scraper_response = MagicMock()
        mock_scraper_response.status_code = 200
        mock_scraper_response.content = b"<html><body>Data</body></html>"
        mock_get.return_value = mock_scraper_response

        app = build_graph()
        final_state = app.invoke(self.initial_state)

        self.assertIn("final_report", final_state)
        self.assertEqual(final_state["final_report"], "Final Markdown Report")
        self.assertEqual(final_state["search_urls"], ["https://testcorp.com"])

if __name__ == '__main__':
    unittest.main()
