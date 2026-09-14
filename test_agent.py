import unittest
from unittest.mock import patch, MagicMock
import os
from corporate_intelligence_agent import search_discovery, deep_scraper, data_extraction_synthesis

class TestCorporateIntelligenceAgent(unittest.TestCase):

    def setUp(self):
        os.environ["TAVILY_API_KEY"] = "dummy_tavily_key"
        os.environ["OPENAI_API_KEY"] = "dummy_openai_key"

    def tearDown(self):
        if "TAVILY_API_KEY" in os.environ:
            del os.environ["TAVILY_API_KEY"]
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]

    @patch('corporate_intelligence_agent.requests.post')
    def test_search_discovery(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [{"url": "https://example.com"}]
        }
        mock_post.return_value = mock_response

        state = {"company_name": "TestCompany"}
        result = search_discovery(state)

        self.assertEqual(result["search_urls"], ["https://example.com"])
        mock_post.assert_called_once()

    @patch('corporate_intelligence_agent.requests.get')
    def test_deep_scraper(self, mock_get):
        mock_response = MagicMock()
        mock_response.content = b"<html><body><p>Test content</p></body></html>"
        mock_get.return_value = mock_response

        state = {"search_urls": ["https://example.com", "https://linkedin.com/test"]}
        result = deep_scraper(state)

        self.assertIn("Source: https://example.com", result["scraped_data"])
        self.assertIn("Content: Test content", result["scraped_data"])
        self.assertNotIn("https://linkedin.com/test", result["scraped_data"])
        mock_get.assert_called_once()

    @patch('corporate_intelligence_agent.requests.post')
    def test_data_extraction_synthesis(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "# Final Report\n## Section"}}]
        }
        mock_post.return_value = mock_response

        state = {"company_name": "TestCompany", "scraped_data": "Test content"}
        result = data_extraction_synthesis(state)

        self.assertEqual(result["final_report"], "# Final Report\n## Section")
        mock_post.assert_called_once()

if __name__ == '__main__':
    unittest.main()
