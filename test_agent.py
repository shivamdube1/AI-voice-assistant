import unittest
from unittest.mock import patch, MagicMock
import os
from corporate_intelligence_agent import search_discovery, deep_scraper, data_extraction_synthesis, AgentState

class TestCorporateIntelligenceAgent(unittest.TestCase):

    @patch('os.environ.get')
    @patch('requests.post')
    def test_search_discovery(self, mock_post, mock_env):
        # Mock environment variables
        mock_env.return_value = 'fake_tavily_api_key'

        # Mock the Tavily API response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"url": "https://example.com/company1"},
                {"url": "https://example.com/company2"}
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        state: AgentState = {
            "company_name": "TestCorp",
            "search_results": [],
            "scraped_texts": [],
            "report": ""
        }

        new_state = search_discovery(state)

        self.assertEqual(len(new_state["search_results"]), 2)
        self.assertEqual(new_state["search_results"][0], "https://example.com/company1")

    @patch('requests.get')
    def test_deep_scraper(self, mock_get):
        # Mock the website response
        mock_response = MagicMock()
        mock_response.text = "<html><body><p>Test Company is great.</p><p>We have 500 employees.</p></body></html>"
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        state: AgentState = {
            "company_name": "TestCorp",
            "search_results": ["https://example.com/company1"],
            "scraped_texts": [],
            "report": ""
        }

        new_state = deep_scraper(state)

        self.assertEqual(len(new_state["scraped_texts"]), 1)
        self.assertIn("Test Company is great.", new_state["scraped_texts"][0])
        self.assertIn("We have 500 employees.", new_state["scraped_texts"][0])

    @patch('os.environ.get')
    @patch('requests.post')
    def test_data_extraction_synthesis(self, mock_post, mock_env):
        # Mock environment variables
        mock_env.return_value = 'fake_openai_api_key'

        # Mock the OpenAI API response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "Executive Summary:\nTestCorp is a great company.\nCompany Size: 500 employees."
                    }
                }
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        state: AgentState = {
            "company_name": "TestCorp",
            "search_results": ["https://example.com/company1"],
            "scraped_texts": ["Test Company is great. We have 500 employees."],
            "report": ""
        }

        new_state = data_extraction_synthesis(state)

        self.assertIn("Executive Summary", new_state["report"])
        self.assertIn("500 employees", new_state["report"])

if __name__ == '__main__':
    unittest.main()
