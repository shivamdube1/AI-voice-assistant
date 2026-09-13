import unittest
import os
from unittest.mock import patch, MagicMock
from corporate_intelligence_agent import (
    search_discovery,
    deep_scraper,
    data_extraction_synthesis,
    create_agent,
    AgentState
)

class TestCorporateIntelligenceAgent(unittest.TestCase):
    def setUp(self):
        # Set dummy environment variables to traverse API code paths
        os.environ["TAVILY_API_KEY"] = "dummy_tavily_key"
        os.environ["OPENAI_API_KEY"] = "dummy_openai_key"

        self.initial_state = AgentState(
            company_name="Test Company",
            search_urls=[],
            scraped_data="",
            final_report=""
        )

    @patch('corporate_intelligence_agent.requests.post')
    def test_search_discovery(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [{"url": "https://testcompany.com"}]
        }
        mock_post.return_value = mock_response

        state = search_discovery(self.initial_state)

        self.assertEqual(state["search_urls"], ["https://testcompany.com"])
        mock_post.assert_called_once()

    @patch('corporate_intelligence_agent.requests.get')
    def test_deep_scraper(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"<html><body>Test Data</body></html>"
        mock_get.return_value = mock_response

        state = AgentState(
            company_name="Test Company",
            search_urls=["https://testcompany.com", "https://linkedin.com/company/test"],
            scraped_data="",
            final_report=""
        )

        new_state = deep_scraper(state)

        self.assertIn("Test Data", new_state["scraped_data"])
        mock_get.assert_called_once() # Should only be called once because linkedin is skipped

    @patch('corporate_intelligence_agent.requests.post')
    def test_data_extraction_synthesis(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Final Generated Report"}}]
        }
        mock_post.return_value = mock_response

        state = AgentState(
            company_name="Test Company",
            search_urls=["https://testcompany.com"],
            scraped_data="Test Data",
            final_report=""
        )

        new_state = data_extraction_synthesis(state)

        self.assertEqual(new_state["final_report"], "Final Generated Report")
        mock_post.assert_called_once()

    @patch('corporate_intelligence_agent.requests.get')
    @patch('corporate_intelligence_agent.requests.post')
    def test_pipeline_execution(self, mock_post, mock_get):
        # Mock search discovery post
        mock_post_search_response = MagicMock()
        mock_post_search_response.json.return_value = {
            "results": [{"url": "https://testcompany.com"}]
        }

        # Mock data extraction synthesis post
        mock_post_llm_response = MagicMock()
        mock_post_llm_response.json.return_value = {
            "choices": [{"message": {"content": "Final E2E Report"}}]
        }

        mock_post.side_effect = [mock_post_search_response, mock_post_llm_response]

        # Mock get
        mock_get_response = MagicMock()
        mock_get_response.status_code = 200
        mock_get_response.content = b"<html><body>Test Data E2E</body></html>"
        mock_get.return_value = mock_get_response

        app = create_agent()
        final_state = None
        for output in app.stream(self.initial_state):
            for key, value in output.items():
                final_state = value

        self.assertIsNotNone(final_state)
        self.assertEqual(final_state["final_report"], "Final E2E Report")
        self.assertEqual(mock_post.call_count, 2)
        mock_get.assert_called_once()

if __name__ == '__main__':
    unittest.main()
