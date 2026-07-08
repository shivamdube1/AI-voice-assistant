import os
import unittest
from unittest.mock import patch, MagicMock
from corporate_intelligence_agent import search_discovery, deep_scraper, data_extraction_synthesis

class TestCorporateIntelligenceAgent(unittest.TestCase):
    def setUp(self):
        # Set dummy env vars so the functions attempt the API paths
        os.environ["TAVILY_API_KEY"] = "dummy_tavily_key"
        os.environ["OPENAI_API_KEY"] = "dummy_openai_key"

    @patch("requests.post")
    def test_search_discovery(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {"url": "https://example.com"},
                {"url": "https://linkedin.com/company/example"}
            ]
        }
        mock_post.return_value = mock_response

        state = {"company_name": "Example Corp"}
        result = search_discovery(state)

        self.assertEqual(len(result["search_urls"]), 2)
        self.assertIn("https://example.com", result["search_urls"])

    @patch("requests.get")
    def test_deep_scraper(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html><body><p>This is Example Corp. We have 500 employees.</p></body></html>"
        mock_get.return_value = mock_response

        state = {"search_urls": ["https://example.com", "https://linkedin.com/company/example"]}
        result = deep_scraper(state)

        # Scraper should skip linkedin.com
        self.assertIn("This is Example Corp. We have 500 employees.", result["scraped_data"])
        self.assertEqual(mock_get.call_count, 1) # Only called for example.com

    @patch("requests.post")
    def test_data_extraction_synthesis(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "# Executive Summary\n\nThis is a mock report."
                    }
                }
            ]
        }
        mock_post.return_value = mock_response

        state = {
            "company_name": "Example Corp",
            "scraped_data": "Source: https://example.com\nThis is Example Corp. We have 500 employees."
        }
        result = data_extraction_synthesis(state)

        self.assertIn("# Executive Summary", result["final_report"])
        self.assertIn("mock report", result["final_report"])

if __name__ == "__main__":
    unittest.main()
