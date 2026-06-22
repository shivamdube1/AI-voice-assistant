import unittest
from unittest.mock import patch, MagicMock
from corporate_intelligence_agent import search_discovery, AgentState
import os

class TestSearchDiscovery(unittest.TestCase):
    @patch('corporate_intelligence_agent.requests.post')
    @patch.dict(os.environ, {"TAVILY_API_KEY": "test_key"})
    def test_search_discovery_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [{"url": "http://example.com"}]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        initial_state: AgentState = {
            "company_name": "TestCorp",
            "search_results": [],
            "scraped_data": "",
            "final_report": ""
        }

        result_state = search_discovery(initial_state)

        # 3 queries are made, each returning 1 result, total 3 URLs expected
        self.assertEqual(len(result_state["search_results"]), 3)
        self.assertEqual(result_state["search_results"][0]["url"], "http://example.com")
        self.assertEqual(mock_post.call_count, 3)

from corporate_intelligence_agent import deep_scraper

class TestDeepScraper(unittest.TestCase):
    @patch('corporate_intelligence_agent.requests.get')
    def test_deep_scraper_success(self, mock_get):
        mock_response = MagicMock()
        mock_response.text = "<html><body><p>This is test content for TestCorp. Employees: 500.</p></body></html>"
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        initial_state: AgentState = {
            "company_name": "TestCorp",
            "search_results": [{"url": "http://example.com"}],
            "scraped_data": "",
            "final_report": ""
        }

        result_state = deep_scraper(initial_state)

        self.assertIn("This is test content for TestCorp. Employees: 500.", result_state["scraped_data"])
        self.assertIn("Source: http://example.com", result_state["scraped_data"])
        mock_get.assert_called_once()


from corporate_intelligence_agent import data_extraction_synthesis

class TestDataExtractionSynthesis(unittest.TestCase):
    @patch('corporate_intelligence_agent.requests.post')
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_openai_key"})
    def test_data_extraction_synthesis_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "# Executive Summary\nTest report content."}}]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        initial_state: AgentState = {
            "company_name": "TestCorp",
            "search_results": [],
            "scraped_data": "Some raw data here.",
            "final_report": ""
        }

        result_state = data_extraction_synthesis(initial_state)

        self.assertEqual(result_state["final_report"], "# Executive Summary\nTest report content.")
        mock_post.assert_called_once()

    @patch.dict(os.environ, clear=True)
    def test_no_api_key(self):
        initial_state: AgentState = {
            "company_name": "TestCorp",
            "search_results": [],
            "scraped_data": "Data",
            "final_report": ""
        }

        result_state = data_extraction_synthesis(initial_state)
        self.assertEqual(result_state["final_report"], "Error: OPENAI_API_KEY not set.")


if __name__ == '__main__':
    unittest.main()
