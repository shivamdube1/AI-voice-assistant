import unittest
from unittest.mock import patch, MagicMock
import os
from corporate_intelligence_agent import app, search_discovery, deep_scraper, data_extraction_synthesis

class TestCorporateIntelligenceAgent(unittest.TestCase):
    def setUp(self):
        # Set dummy environment variables to ensure API code paths are tested
        os.environ["TAVILY_API_KEY"] = "dummy_tavily_key"
        os.environ["OPENAI_API_KEY"] = "dummy_openai_key"

    def tearDown(self):
        # Clean up
        if "TAVILY_API_KEY" in os.environ:
            del os.environ["TAVILY_API_KEY"]
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]

    @patch('corporate_intelligence_agent.requests.post')
    def test_search_discovery_with_api_key(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [{"url": "https://www.example.com"}, {"url": "https://en.wikipedia.org/wiki/Example"}]
        }
        mock_post.return_value = mock_response

        state = {"company_name": "Example Corp"}
        result = search_discovery(state)

        self.assertIn("search_urls", result)
        self.assertEqual(len(result["search_urls"]), 2)
        self.assertEqual(result["search_urls"][0], "https://www.example.com")

    @patch('corporate_intelligence_agent.requests.get')
    def test_deep_scraper(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"<html><body><p>This is test content for Example Corp.</p></body></html>"
        mock_get.return_value = mock_response

        state = {"search_urls": ["https://www.example.com", "https://www.linkedin.com/company/example"]}
        result = deep_scraper(state)

        self.assertIn("scraped_data", result)
        self.assertIn("This is test content for Example Corp.", result["scraped_data"])
        # Should only call requests.get once because linkedin is skipped
        self.assertEqual(mock_get.call_count, 1)

    @patch('corporate_intelligence_agent.requests.post')
    def test_data_extraction_synthesis_with_api_key(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "# Executive Summary\nTest report content."}}]
        }
        mock_post.return_value = mock_response

        state = {"company_name": "Example Corp", "scraped_data": "Some scraped data."}
        result = data_extraction_synthesis(state)

        self.assertIn("final_report", result)
        self.assertIn("Test report content", result["final_report"])

if __name__ == '__main__':
    unittest.main()
