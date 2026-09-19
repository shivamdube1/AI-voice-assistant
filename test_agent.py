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
        # Set dummy environment variables to trigger API code paths instead of fallbacks
        os.environ["TAVILY_API_KEY"] = "dummy_tavily_key"
        os.environ["OPENAI_API_KEY"] = "dummy_openai_key"

    def tearDown(self):
        # Clean up dummy environment variables
        if "TAVILY_API_KEY" in os.environ:
            del os.environ["TAVILY_API_KEY"]
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]

    @patch('requests.post')
    def test_search_discovery(self, mock_post):
        # Setup mock response for Tavily API
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"url": "https://www.example.com"},
                {"url": "https://www.linkedin.com/company/example"}
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        state: AgentState = {"company_name": "Example Corp"}
        new_state = search_discovery(state)

        self.assertIn("search_urls", new_state)
        self.assertEqual(len(new_state["search_urls"]), 2)
        self.assertIn("https://www.example.com", new_state["search_urls"])
        mock_post.assert_called_once()

    @patch('requests.get')
    def test_deep_scraper(self, mock_get):
        # Setup mock response for requests.get
        mock_response = MagicMock()
        mock_response.text = "<html><body><h1>Example Corp</h1><p>We are a cool company.</p></body></html>"
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        state: AgentState = {
            "search_urls": [
                "https://www.example.com",
                "https://www.linkedin.com/company/example" # This should be skipped
            ]
        }
        new_state = deep_scraper(state)

        self.assertIn("scraped_data", new_state)
        self.assertIn("Example Corp", new_state["scraped_data"])
        self.assertIn("We are a cool company.", new_state["scraped_data"])
        # Should only be called once because the linkedin URL is skipped
        mock_get.assert_called_once()

    @patch('requests.post')
    def test_data_extraction_synthesis(self, mock_post):
        # Setup mock response for OpenAI API
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "# Test Report\n\n## Executive Summary\nTest company is testy."
                    }
                }
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        state: AgentState = {
            "company_name": "Example Corp",
            "scraped_data": "Lots of raw data here."
        }
        new_state = data_extraction_synthesis(state)

        self.assertIn("final_report", new_state)
        self.assertIn("# Test Report", new_state["final_report"])
        mock_post.assert_called_once()

if __name__ == '__main__':
    unittest.main()
