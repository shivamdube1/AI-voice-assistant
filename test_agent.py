import unittest
from unittest.mock import patch, MagicMock
import os
from corporate_intelligence_agent import (
    search_discovery,
    deep_scraper,
    data_extraction_synthesis,
    AgentState
)

class TestCorporateIntelligenceAgent(unittest.TestCase):
    def setUp(self):
        # Set dummy environment variables to ensure we hit the API paths, not the fallback paths
        os.environ["TAVILY_API_KEY"] = "dummy_tavily_key"
        os.environ["OPENAI_API_KEY"] = "dummy_openai_key"

    def tearDown(self):
        # Clean up
        if "TAVILY_API_KEY" in os.environ:
            del os.environ["TAVILY_API_KEY"]
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]

    @patch('corporate_intelligence_agent.requests.post')
    def test_search_discovery(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"url": "https://example.com"},
                {"url": "https://example.com/news"}
            ]
        }
        mock_post.return_value = mock_response

        initial_state = AgentState(company_name="TestCorp", search_urls=[], scraped_data="", final_report="")
        new_state = search_discovery(initial_state)

        self.assertEqual(len(new_state["search_urls"]), 2)
        self.assertEqual(new_state["search_urls"][0], "https://example.com")
        self.assertEqual(new_state["search_urls"][1], "https://example.com/news")

    @patch('corporate_intelligence_agent.requests.get')
    def test_deep_scraper(self, mock_get):
        # Create a mock response
        mock_response = MagicMock()
        mock_response.content = b"<html><body><p>This is a test corp about us page.</p></body></html>"
        mock_get.return_value = mock_response

        state = AgentState(
            company_name="TestCorp",
            search_urls=["https://example.com", "https://linkedin.com/company/testcorp"],
            scraped_data="",
            final_report=""
        )

        new_state = deep_scraper(state)

        # Verify that only the non-linkedin URL was scraped
        mock_get.assert_called_once_with("https://example.com", headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}, timeout=10)
        self.assertIn("This is a test corp about us page.", new_state["scraped_data"])
        self.assertIn("Source: https://example.com", new_state["scraped_data"])
    @patch('corporate_intelligence_agent.requests.post')
    def test_data_extraction_synthesis(self, mock_post):
        mock_response = MagicMock()
        mock_report = "# Corporate Intelligence Report\nTestCorp is a testing company."
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": mock_report
                    }
                }
            ]
        }
        mock_post.return_value = mock_response

        state = AgentState(
            company_name="TestCorp",
            search_urls=[],
            scraped_data="This is a test corp about us page.",
            final_report=""
        )

        new_state = data_extraction_synthesis(state)

        self.assertEqual(new_state["final_report"], mock_report)
        mock_post.assert_called_once()

if __name__ == '__main__':
    unittest.main()
