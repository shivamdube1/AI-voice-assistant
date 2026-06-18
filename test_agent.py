import unittest
from unittest.mock import patch, MagicMock
import os
import json
from corporate_intelligence_agent import search_discovery, deep_scraper, data_extraction_synthesis

class TestCorporateIntelligenceAgent(unittest.TestCase):
    def setUp(self):
        os.environ["TAVILY_API_KEY"] = "fake_tavily_key"
        os.environ["OPENAI_API_KEY"] = "fake_openai_key"

    @patch("corporate_intelligence_agent.requests.post")
    def test_search_discovery(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"url": "https://example.com"},
                {"url": "https://example.com/about"}
            ]
        }
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        state = {"company_name": "TestCorp", "urls_to_scrape": [], "scraped_data": "", "final_report": ""}
        result_state = search_discovery(state)

        self.assertEqual(result_state["urls_to_scrape"], ["https://example.com", "https://example.com/about"])
        mock_post.assert_called_once()

    @patch("corporate_intelligence_agent.requests.get")
    def test_deep_scraper(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html><body><p>TestCorp has 500 employees and made $10M in revenue.</p></body></html>"
        mock_get.return_value = mock_response

        state = {
            "company_name": "TestCorp",
            "urls_to_scrape": ["https://example.com"],
            "scraped_data": "",
            "final_report": ""
        }
        result_state = deep_scraper(state)

        self.assertIn("TestCorp has 500 employees and made $10M in revenue.", result_state["scraped_data"])
        mock_get.assert_called_once()

    @patch("corporate_intelligence_agent.requests.post")
    def test_data_extraction_synthesis(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "### Executive Summary\nTestCorp overview."
                    }
                }
            ]
        }
        mock_post.return_value = mock_response

        state = {
            "company_name": "TestCorp",
            "urls_to_scrape": ["https://example.com"],
            "scraped_data": "Source: https://example.com\nContent: TestCorp has 500 employees and made $10M in revenue.",
            "final_report": ""
        }
        result_state = data_extraction_synthesis(state)

        self.assertIn("Executive Summary", result_state["final_report"])
        mock_post.assert_called_once()

if __name__ == "__main__":
    unittest.main()
