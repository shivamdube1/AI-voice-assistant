import unittest
from unittest.mock import patch, MagicMock
import os
from corporate_intelligence_agent import app

class TestCorporateIntelligenceAgent(unittest.TestCase):

    def setUp(self):
        # Set dummy environment variables to test API paths
        os.environ["TAVILY_API_KEY"] = "dummy_tavily_key"
        os.environ["OPENAI_API_KEY"] = "dummy_openai_key"

    def tearDown(self):
        if "TAVILY_API_KEY" in os.environ:
            del os.environ["TAVILY_API_KEY"]
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]

    @patch('corporate_intelligence_agent.requests.post')
    @patch('corporate_intelligence_agent.requests.get')
    def test_pipeline_execution(self, mock_get, mock_post):
        # Mock Tavily search response
        mock_tavily_response = MagicMock()
        mock_tavily_response.json.return_value = {
            "results": [
                {"url": "https://example.com/about"},
                {"url": "https://linkedin.com/company/example"} # Should be skipped
            ]
        }
        mock_tavily_response.raise_for_status.return_value = None

        # Mock OpenAI response
        mock_openai_response = MagicMock()
        mock_openai_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "Executive Summary: Dummy report.\n\nCompany Profile: Dummy specialization."
                    }
                }
            ]
        }
        mock_openai_response.raise_for_status.return_value = None

        # Configure requests.post to return mock_tavily_response then mock_openai_response
        mock_post.side_effect = [mock_tavily_response, mock_openai_response]

        # Mock requests.get for BeautifulSoup scraping
        mock_get_response = MagicMock()
        mock_get_response.status_code = 200
        mock_get_response.content = b"<html><body>Dummy scraped text containing employees and revenue.</body></html>"
        mock_get.return_value = mock_get_response

        initial_state = {
            "company_name": "TestCompany",
            "search_urls": [],
            "scraped_data": "",
            "final_report": ""
        }

        final_state = None
        for output in app.stream(initial_state):
            for key, value in output.items():
                final_state = value

        self.assertIsNotNone(final_state)

        # Verify Tavily results were parsed and stored
        self.assertIn("https://example.com/about", final_state["search_urls"])
        self.assertIn("https://linkedin.com/company/example", final_state["search_urls"])

        # Verify scraped data contains text from mock_get
        self.assertIn("Dummy scraped text containing employees and revenue.", final_state["scraped_data"])

        # Verify the final report
        self.assertIn("Executive Summary: Dummy report.", final_state["final_report"])

if __name__ == '__main__':
    unittest.main()
