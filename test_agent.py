import unittest
from unittest.mock import patch, MagicMock
import os
from corporate_intelligence_agent import (
    search_discovery,
    deep_scraper,
    data_extraction_synthesis,
    build_agent_graph
)

class TestCorporateIntelligenceAgent(unittest.TestCase):
    def setUp(self):
        # Set dummy API keys to traverse the API paths instead of fallbacks
        os.environ["TAVILY_API_KEY"] = "dummy_tavily_key"
        os.environ["OPENAI_API_KEY"] = "dummy_openai_key"

    @patch("corporate_intelligence_agent.requests.post")
    def test_search_discovery(self, mock_post):
        # Mock Tavily response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"url": "https://example-company.com"},
                {"url": "https://linkedin.com/company/example"}
            ]
        }
        mock_post.return_value = mock_response

        state = {"company_name": "Example Corp"}
        result = search_discovery(state)

        self.assertIn("search_urls", result)
        self.assertEqual(len(result["search_urls"]), 2)
        self.assertIn("https://example-company.com", result["search_urls"])

    @patch("corporate_intelligence_agent.requests.get")
    def test_deep_scraper(self, mock_get):
        # Mock website response
        mock_response = MagicMock()
        mock_response.text = "<html><body><h1>About Us</h1><p>We are a great company with 500 employees.</p></body></html>"
        mock_get.return_value = mock_response

        state = {
            "company_name": "Example Corp",
            "search_urls": [
                "https://example-company.com",
                "https://linkedin.com/company/example"  # This should be skipped
            ]
        }
        result = deep_scraper(state)

        self.assertIn("scraped_data", result)
        # Should contain text from example-company.com
        self.assertIn("About Us", result["scraped_data"])
        self.assertIn("500 employees", result["scraped_data"])
        # Should have skipped LinkedIn, so get was only called once
        mock_get.assert_called_once_with("https://example-company.com", headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}, timeout=10)

    @patch("corporate_intelligence_agent.requests.post")
    def test_data_extraction_synthesis(self, mock_post):
        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "# Executive Summary\n\nExample Corp is a great company.\n\n## Company Size\n500 employees."
                    }
                }
            ]
        }
        mock_post.return_value = mock_response

        state = {
            "company_name": "Example Corp",
            "scraped_data": "--- Data from https://example-company.com ---\nAbout Us We are a great company with 500 employees."
        }
        result = data_extraction_synthesis(state)

        self.assertIn("final_report", result)
        self.assertIn("# Executive Summary", result["final_report"])
        self.assertIn("500 employees", result["final_report"])

    @patch("corporate_intelligence_agent.requests.get")
    @patch("corporate_intelligence_agent.requests.post")
    def test_full_pipeline(self, mock_post, mock_get):
        # We need mock_post to return different things based on the URL
        def mock_post_side_effect(url, **kwargs):
            mock_response = MagicMock()
            if "tavily" in url:
                mock_response.json.return_value = {
                    "results": [{"url": "https://example.com"}]
                }
            elif "openai" in url:
                mock_response.json.return_value = {
                    "choices": [{"message": {"content": "Final Report Mock"}}]
                }
            return mock_response

        mock_post.side_effect = mock_post_side_effect

        # Mock get for deep_scraper
        mock_get_response = MagicMock()
        mock_get_response.text = "<p>Data</p>"
        mock_get.return_value = mock_get_response

        graph = build_agent_graph()
        initial_state = {"company_name": "Test Company"}
        final_state = graph.invoke(initial_state)

        self.assertIn("final_report", final_state)
        self.assertEqual(final_state["final_report"], "Final Report Mock")
        self.assertIn("search_urls", final_state)
        self.assertIn("scraped_data", final_state)

if __name__ == '__main__':
    unittest.main()
