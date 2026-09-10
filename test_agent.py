import unittest
from unittest.mock import patch, MagicMock
import os
from corporate_intelligence_agent import search_discovery, deep_scraper, data_extraction_synthesis, AgentState, build_graph

class TestCorporateIntelligenceAgent(unittest.TestCase):
    def setUp(self):
        # Set dummy environment variables to ensure we hit the API code paths
        os.environ["TAVILY_API_KEY"] = "dummy_tavily_key"
        os.environ["OPENAI_API_KEY"] = "dummy_openai_key"

    def tearDown(self):
        # Clean up environment variables
        if "TAVILY_API_KEY" in os.environ:
            del os.environ["TAVILY_API_KEY"]
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]

    @patch("corporate_intelligence_agent.requests.post")
    def test_search_discovery(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"url": "https://example.com"},
                {"url": "https://linkedin.com/company/example"}
            ]
        }
        mock_post.return_value = mock_response

        initial_state = AgentState(
            company_name="ExampleCorp",
            search_urls=[],
            scraped_data="",
            final_report=""
        )

        result_state = search_discovery(initial_state)
        self.assertIn("https://example.com", result_state["search_urls"])
        self.assertIn("https://linkedin.com/company/example", result_state["search_urls"])

    @patch("corporate_intelligence_agent.requests.get")
    def test_deep_scraper_skips_linkedin(self, mock_get):
        mock_response = MagicMock()
        mock_response.text = "<html><body><h1>Hello World</h1></body></html>"
        mock_get.return_value = mock_response

        initial_state = AgentState(
            company_name="ExampleCorp",
            search_urls=["https://example.com", "https://linkedin.com/company/example"],
            scraped_data="",
            final_report=""
        )

        result_state = deep_scraper(initial_state)

        # Verify get was only called once (for example.com)
        mock_get.assert_called_once_with("https://example.com", timeout=10)
        self.assertIn("Hello World", result_state["scraped_data"])

    @patch("corporate_intelligence_agent.requests.post")
    def test_data_extraction_synthesis(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "# Executive Summary\nTest Report"
                    }
                }
            ]
        }
        mock_post.return_value = mock_response

        initial_state = AgentState(
            company_name="ExampleCorp",
            search_urls=[],
            scraped_data="Some data here",
            final_report=""
        )

        result_state = data_extraction_synthesis(initial_state)

        self.assertIn("Test Report", result_state["final_report"])

        # Verify OpenAI API was called
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertEqual(args[0], "https://api.openai.com/v1/chat/completions")
