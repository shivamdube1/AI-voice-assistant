import unittest
import os
from unittest.mock import patch, MagicMock
from corporate_intelligence_agent import (
    search_discovery,
    deep_scraper,
    data_extraction_synthesis,
    build_graph,
    AgentState
)

class TestCorporateIntelligenceAgent(unittest.TestCase):
    def setUp(self):
        # Set dummy environment variables to ensure API code paths are traversed
        os.environ["TAVILY_API_KEY"] = "dummy_tavily_key"
        os.environ["OPENAI_API_KEY"] = "dummy_openai_key"
        self.company_name = "TestCompany"

    @patch('corporate_intelligence_agent.requests.post')
    def test_search_discovery(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"url": "https://testcompany.com"},
                {"url": "https://testcompany.com/news"}
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        state = {"company_name": self.company_name}
        result = search_discovery(state)

        self.assertIn("search_urls", result)
        self.assertEqual(len(result["search_urls"]), 2)
        self.assertEqual(result["search_urls"][0], "https://testcompany.com")
        mock_post.assert_called_once()

    @patch('corporate_intelligence_agent.requests.get')
    def test_deep_scraper(self, mock_get):
        mock_response = MagicMock()
        mock_response.text = "<html><body><p>Test company data</p></body></html>"
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        state = {"search_urls": ["https://testcompany.com", "https://linkedin.com/company/test"]}
        result = deep_scraper(state)

        self.assertIn("scraped_data", result)
        self.assertIn("Test company data", result["scraped_data"])

        # linkedin url should be skipped
        mock_get.assert_called_once_with("https://testcompany.com", headers=unittest.mock.ANY, timeout=10)

    @patch('corporate_intelligence_agent.requests.post')
    def test_data_extraction_synthesis(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {"message": {"content": "# Final Report\nTest company is great."}}
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        state = {
            "company_name": self.company_name,
            "scraped_data": "Test data"
        }
        result = data_extraction_synthesis(state)

        self.assertIn("final_report", result)
        self.assertIn("Final Report", result["final_report"])
        mock_post.assert_called_once()

    @patch('corporate_intelligence_agent.requests.post')
    @patch('corporate_intelligence_agent.requests.get')
    def test_full_graph_execution(self, mock_get, mock_post):
        # Setup mock for search_discovery (post)
        search_response = MagicMock()
        search_response.json.return_value = {
            "results": [{"url": "https://testcompany.com"}]
        }
        search_response.raise_for_status.return_value = None

        # Setup mock for data_extraction_synthesis (post)
        llm_response = MagicMock()
        llm_response.json.return_value = {
            "choices": [{"message": {"content": "# Final Output"}}]
        }
        llm_response.raise_for_status.return_value = None

        # mock_post is called twice, return different responses
        mock_post.side_effect = [search_response, llm_response]

        # Setup mock for deep_scraper (get)
        scrape_response = MagicMock()
        scrape_response.text = "<html><body>Data</body></html>"
        scrape_response.raise_for_status.return_value = None
        mock_get.return_value = scrape_response

        app = build_graph()
        initial_state = {"company_name": self.company_name}

        final_state = None
        for output in app.stream(initial_state):
            for key, value in output.items():
                final_state = value

        self.assertIsNotNone(final_state)
        self.assertIn("final_report", final_state)
        self.assertEqual(final_state["final_report"], "# Final Output")

if __name__ == '__main__':
    unittest.main()
