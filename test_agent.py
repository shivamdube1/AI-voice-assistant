import unittest
import os
from unittest.mock import patch, MagicMock
from corporate_intelligence_agent import search_discovery, deep_scraper, data_extraction_synthesis

class TestCorporateIntelligenceAgent(unittest.TestCase):
    def setUp(self):
        os.environ["TAVILY_API_KEY"] = "dummy_tavily_key"
        os.environ["OPENAI_API_KEY"] = "dummy_openai_key"

    @patch("corporate_intelligence_agent.requests.post")
    def test_search_discovery(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"url": "https://example.com"},
                {"url": "https://news.example.com"}
            ]
        }
        mock_post.return_value = mock_response

        state = {"company_name": "TestCorp"}
        result_state = search_discovery(state)

        self.assertIn("search_urls", result_state)
        self.assertEqual(result_state["search_urls"], ["https://example.com", "https://news.example.com"])
        mock_post.assert_called_once()

    @patch("corporate_intelligence_agent.requests.get")
    def test_deep_scraper(self, mock_get):
        mock_response = MagicMock()
        mock_response.text = "<html><body><p>TestCorp is a leading AI company.</p></body></html>"
        mock_get.return_value = mock_response

        state = {"search_urls": ["https://example.com", "https://linkedin.com/testcorp"]}
        result_state = deep_scraper(state)

        self.assertIn("scraped_data", result_state)
        self.assertIn("TestCorp is a leading AI company.", result_state["scraped_data"])
        # Should only be called once because linkedin.com is skipped
        mock_get.assert_called_once()

    @patch("corporate_intelligence_agent.requests.post")
    def test_data_extraction_synthesis(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "# Executive Summary\nTestCorp is a great company."
                    }
                }
            ]
        }
        mock_post.return_value = mock_response

        state = {
            "company_name": "TestCorp",
            "scraped_data": "TestCorp is a leading AI company."
        }
        result_state = data_extraction_synthesis(state)

        self.assertIn("final_report", result_state)
        self.assertIn("Executive Summary", result_state["final_report"])
        self.assertIn("TestCorp is a great company.", result_state["final_report"])
        mock_post.assert_called_once()

if __name__ == "__main__":
    unittest.main()
