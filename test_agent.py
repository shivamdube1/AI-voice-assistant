import unittest
from unittest.mock import patch, MagicMock
from corporate_intelligence_agent import (
    search_discovery,
    deep_scraper,
    data_extraction_synthesis,
    AgentState
)
import os

class TestCorporateIntelligenceAgent(unittest.TestCase):
    @patch('os.environ.get')
    @patch('requests.post')
    def test_search_discovery(self, mock_post, mock_env_get):
        # Mock environment variable
        mock_env_get.return_value = 'fake_tavily_api_key'

        # Mock Tavily response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"url": "https://example.com"},
                {"url": "https://linkedin.com/company/example"}
            ]
        }
        mock_post.return_value = mock_response

        state = AgentState(company_name="Example Corp", search_urls=[], scraped_data="", final_report="")
        new_state = search_discovery(state)

        self.assertEqual(new_state["search_urls"], ["https://example.com", "https://linkedin.com/company/example"])
        mock_post.assert_called_once()

    @patch('requests.get')
    def test_deep_scraper(self, mock_get):
        # Mock requests.get response
        mock_response = MagicMock()
        mock_response.content = b"<html><body><p>We have 500 employees.</p><script>ignore me</script></body></html>"
        mock_get.return_value = mock_response

        state = AgentState(company_name="Example Corp", search_urls=["https://example.com"], scraped_data="", final_report="")
        new_state = deep_scraper(state)

        self.assertIn("We have 500 employees.", new_state["scraped_data"])
        self.assertNotIn("ignore me", new_state["scraped_data"])
        mock_get.assert_called_once()

    @patch('os.environ.get')
    @patch('requests.post')
    def test_data_extraction_synthesis(self, mock_post, mock_env_get):
        # Mock environment variable
        mock_env_get.return_value = 'fake_openai_api_key'

        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "# Markdown Report\nInsufficient data found for this metric."
                    }
                }
            ]
        }
        mock_post.return_value = mock_response

        state = AgentState(company_name="Example Corp", search_urls=[], scraped_data="Some raw text", final_report="")
        new_state = data_extraction_synthesis(state)

        self.assertIn("Insufficient data found for this metric", new_state["final_report"])
        mock_post.assert_called_once()

if __name__ == "__main__":
    unittest.main()
