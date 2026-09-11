import unittest
from unittest.mock import patch, MagicMock
import os
from corporate_intelligence_agent import search_discovery, deep_scraper, data_extraction_synthesis, AgentState

class TestCorporateIntelligenceAgent(unittest.TestCase):
    def setUp(self):
        # Set dummy environment variables to traverse API code paths
        os.environ["TAVILY_API_KEY"] = "dummy_tavily_key"
        os.environ["OPENAI_API_KEY"] = "dummy_openai_key"
        self.state: AgentState = {"company_name": "TestCompany", "search_urls": [], "scraped_data": "", "final_report": ""}

    @patch("corporate_intelligence_agent.requests.post")
    def test_search_discovery(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [{"url": "https://www.testcompany.com"}, {"url": "https://www.linkedin.com/company/testcompany"}]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = search_discovery(self.state)

        self.assertIn("search_urls", result)
        self.assertEqual(len(result["search_urls"]), 2)
        self.assertEqual(result["search_urls"][0], "https://www.testcompany.com")
        self.assertEqual(result["search_urls"][1], "https://www.linkedin.com/company/testcompany")
        mock_post.assert_called_once()

    @patch("corporate_intelligence_agent.requests.get")
    def test_deep_scraper(self, mock_get):
        self.state["search_urls"] = ["https://www.testcompany.com", "https://www.linkedin.com/company/testcompany"]

        mock_response = MagicMock()
        mock_response.text = "<html><body><p>This is test company data.</p></body></html>"
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = deep_scraper(self.state)

        self.assertIn("scraped_data", result)
        self.assertIn("This is test company data.", result["scraped_data"])
        self.assertIn("Source: https://www.testcompany.com", result["scraped_data"])

        # Ensure linkedin is skipped (mock_get should only be called once)
        self.assertEqual(mock_get.call_count, 1)

    @patch("corporate_intelligence_agent.requests.post")
    def test_data_extraction_synthesis(self, mock_post):
        self.state["scraped_data"] = "Source: https://www.testcompany.com\nContent: This is test company data."

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "# Corporate Intelligence Report\n\n## Executive Summary\nA test company."}}]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = data_extraction_synthesis(self.state)

        self.assertIn("final_report", result)
        self.assertIn("# Corporate Intelligence Report", result["final_report"])
        mock_post.assert_called_once()
