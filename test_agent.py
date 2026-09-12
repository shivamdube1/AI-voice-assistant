import unittest
from unittest.mock import patch, MagicMock
import os
from corporate_intelligence_agent import search_discovery, deep_scraper, data_extraction_synthesis, AgentState

class TestCorporateIntelligenceAgent(unittest.TestCase):
    def setUp(self):
        # Set up dummy environment variables for tests
        os.environ["TAVILY_API_KEY"] = "dummy_tavily_key"
        os.environ["OPENAI_API_KEY"] = "dummy_openai_key"

    def tearDown(self):
        # Clean up environment variables
        if "TAVILY_API_KEY" in os.environ:
            del os.environ["TAVILY_API_KEY"]
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]

    @patch("corporate_intelligence_agent.requests.post")
    def test_search_discovery_success(self, mock_post):
        # Mock Tavily response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {"url": "https://example.com"},
                {"url": "https://example.com/about"}
            ]
        }
        mock_post.return_value = mock_response

        state = {"company_name": "ExampleCorp"}
        new_state = search_discovery(state)

        self.assertEqual(new_state["company_name"], "ExampleCorp")
        self.assertEqual(new_state["search_urls"], ["https://example.com", "https://example.com/about"])
        mock_post.assert_called_once()

    @patch("corporate_intelligence_agent.requests.get")
    def test_deep_scraper_success(self, mock_get):
        # Mock webpage response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"<html><body><p>This is test data about the company.</p></body></html>"
        mock_get.return_value = mock_response

        state = {"company_name": "ExampleCorp", "search_urls": ["https://example.com"]}
        new_state = deep_scraper(state)

        self.assertIn("This is test data about the company.", new_state["scraped_data"])
        self.assertEqual(new_state["company_name"], "ExampleCorp")
        self.assertEqual(new_state["search_urls"], ["https://example.com"])
        mock_get.assert_called_once()

    @patch("corporate_intelligence_agent.requests.post")
    def test_data_extraction_synthesis_success(self, mock_post):
        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "# Corporate Intelligence Report: ExampleCorp\n\n## Executive Summary\nA mock summary."
                    }
                }
            ]
        }
        mock_post.return_value = mock_response

        state = {"company_name": "ExampleCorp", "scraped_data": "Some scraped data."}
        new_state = data_extraction_synthesis(state)

        self.assertIn("# Corporate Intelligence Report: ExampleCorp", new_state["final_report"])
        self.assertEqual(new_state["company_name"], "ExampleCorp")
        mock_post.assert_called_once()

if __name__ == '__main__':
    unittest.main()
