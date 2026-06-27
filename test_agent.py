import unittest
from unittest.mock import patch, MagicMock
from corporate_intelligence_agent import search_discovery, deep_scraper, data_extraction_synthesis

class TestAgent(unittest.TestCase):

    @patch('corporate_intelligence_agent.requests.post')
    def test_search_discovery(self, mock_post):
        # Mock Tavily API response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"url": "https://example.com"},
                {"url": "https://example.com/linkedin"}
            ]
        }
        mock_post.return_value = mock_response

        state = {"company_name": "TestCompany"}
        result = search_discovery(state)

        self.assertIn("search_urls", result)
        self.assertEqual(len(result["search_urls"]), 2)
        self.assertEqual(result["search_urls"][0], "https://example.com")

    @patch('corporate_intelligence_agent.requests.get')
    def test_deep_scraper(self, mock_get):
        # Mock website response
        mock_response = MagicMock()
        mock_response.text = "<html><body><p>Test Company is a great place to work.</p></body></html>"
        mock_get.return_value = mock_response

        state = {"search_urls": ["https://example.com"]}
        result = deep_scraper(state)

        self.assertIn("scraped_data", result)
        self.assertIn("Test Company is a great place to work.", result["scraped_data"])
        self.assertIn("https://example.com", result["scraped_data"])

    @patch('corporate_intelligence_agent.requests.post')
    def test_data_extraction_synthesis(self, mock_post):
        # Mock OpenAI API response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {"message": {"content": "# Test Report\n\nExecutive Summary: TestCompany is an amazing test."}}
            ]
        }
        mock_post.return_value = mock_response

        state = {
            "company_name": "TestCompany",
            "scraped_data": "Source: https://example.com\nContent: Test Company is a great place to work."
        }
        result = data_extraction_synthesis(state)

        self.assertIn("final_report", result)
        self.assertEqual(result["final_report"], "# Test Report\n\nExecutive Summary: TestCompany is an amazing test.")

if __name__ == '__main__':
    unittest.main()
