import unittest
from unittest.mock import patch, MagicMock
import os
from corporate_intelligence_agent import search_discovery, deep_scraper, data_extraction_synthesis, build_graph

class TestCorporateIntelligenceAgent(unittest.TestCase):
    def setUp(self):
        # Set dummy API keys for testing
        os.environ["TAVILY_API_KEY"] = "dummy_tavily_key"
        os.environ["OPENAI_API_KEY"] = "dummy_openai_key"
        self.initial_state = {
            "company_name": "TestCompany",
            "search_urls": [],
            "scraped_data": "",
            "final_report": ""
        }

    @patch('requests.post')
    def test_search_discovery(self, mock_post):
        # Mock Tavily response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"url": "https://testcompany.com"},
                {"url": "https://linkedin.com/company/testcompany"}
            ]
        }
        mock_post.return_value = mock_response

        state = search_discovery(self.initial_state)
        self.assertEqual(len(state["search_urls"]), 2)
        self.assertIn("https://testcompany.com", state["search_urls"])

    @patch('requests.get')
    def test_deep_scraper(self, mock_get):
        # Mock HTML response
        mock_response = MagicMock()
        mock_response.text = "<html><body><p>Welcome to TestCompany.</p><p>We have 500 employees.</p></body></html>"
        mock_get.return_value = mock_response

        state = {"search_urls": ["https://testcompany.com"]}
        result = deep_scraper(state)

        self.assertIn("Welcome to TestCompany.", result["scraped_data"])
        self.assertIn("We have 500 employees.", result["scraped_data"])

    @patch('requests.post')
    def test_data_extraction_synthesis(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "Executive Summary: TestCompany is a tech firm.\nCompany Size: 500"
                    }
                }
            ]
        }
        mock_post.return_value = mock_response

        state = {
            "company_name": "TestCompany",
            "scraped_data": "Welcome to TestCompany. We have 500 employees."
        }
        result = data_extraction_synthesis(state)

        self.assertIn("Executive Summary", result["final_report"])
        self.assertIn("500", result["final_report"])

    @patch('requests.get')
    @patch('requests.post')
    def test_workflow_execution(self, mock_post, mock_get):
        # Mock Tavily response (first post call) and OpenAI response (second post call)
        mock_post.side_effect = [
            MagicMock(json=lambda: {"results": [{"url": "https://testcompany.com"}]}),
            MagicMock(json=lambda: {"choices": [{"message": {"content": "Final Report Mocked"}}]})
        ]

        # Mock Scraper response
        mock_get.return_value = MagicMock(text="<html><body>Test Data</body></html>")

        graph = build_graph()
        result = graph.invoke(self.initial_state)

        self.assertEqual(result["company_name"], "TestCompany")
        self.assertEqual(len(result["search_urls"]), 1)
        self.assertIn("Test Data", result["scraped_data"])
        self.assertEqual(result["final_report"], "Final Report Mocked")

if __name__ == '__main__':
    unittest.main()
