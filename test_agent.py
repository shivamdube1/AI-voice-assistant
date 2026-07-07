import unittest
from unittest.mock import patch, MagicMock
from corporate_intelligence_agent import AgentState, search_discovery, deep_scraper, data_extraction_synthesis
import os

class TestCorporateIntelligenceAgent(unittest.TestCase):
    def setUp(self):
        self.initial_state: AgentState = {
            "company_name": "TestCompany",
            "search_urls": [],
            "scraped_data": "",
            "final_report": ""
        }
        # Set dummy API keys to force execution down the API paths
        os.environ["TAVILY_API_KEY"] = "test_tavily_key"
        os.environ["OPENAI_API_KEY"] = "test_openai_key"

    @patch('corporate_intelligence_agent.requests.post')
    def test_search_discovery(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"url": "https://www.testcompany.com"},
                {"url": "https://www.linkedin.com/company/testcompany"}
            ]
        }
        mock_post.return_value = mock_response

        state = search_discovery(self.initial_state)

        self.assertIn("https://www.testcompany.com", state["search_urls"])
        self.assertIn("https://www.linkedin.com/company/testcompany", state["search_urls"])
        mock_post.assert_called_once()

    @patch('corporate_intelligence_agent.requests.get')
    def test_deep_scraper(self, mock_get):
        mock_response = MagicMock()
        mock_response.text = "<html><body><p>This is a test paragraph with 500 employees.</p></body></html>"
        mock_get.return_value = mock_response

        state = self.initial_state.copy()
        state["search_urls"] = ["https://www.testcompany.com", "https://www.linkedin.com/company/testcompany"]

        new_state = deep_scraper(state)

        self.assertIn("This is a test paragraph", new_state["scraped_data"])
        # ensure linkedin was skipped
        self.assertEqual(mock_get.call_count, 1)

    @patch('corporate_intelligence_agent.requests.post')
    def test_data_extraction_synthesis(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "# TestCompany Report\n\n## Executive Summary\nTestCompany is a test."
                    }
                }
            ]
        }
        mock_post.return_value = mock_response

        state = self.initial_state.copy()
        state["scraped_data"] = "Source: This is a test paragraph with 500 employees."

        new_state = data_extraction_synthesis(state)

        self.assertIn("# TestCompany Report", new_state["final_report"])
        mock_post.assert_called_once()

if __name__ == '__main__':
    unittest.main()
