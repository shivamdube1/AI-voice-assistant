import unittest
import os
from unittest.mock import patch, MagicMock
from corporate_intelligence_agent import app, search_discovery, deep_scraper, data_extraction_synthesis

class TestCorporateIntelligenceAgent(unittest.TestCase):
    def setUp(self):
        # Set dummy environment variables to bypass the missing key checks
        os.environ["TAVILY_API_KEY"] = "dummy_tavily_key"
        os.environ["OPENAI_API_KEY"] = "dummy_openai_key"

    def tearDown(self):
        # Clean up environment variables
        if "TAVILY_API_KEY" in os.environ:
            del os.environ["TAVILY_API_KEY"]
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]

    @patch('corporate_intelligence_agent.requests.post')
    def test_search_discovery(self, mock_post):
        # Mock the Tavily API response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"url": "https://www.testcompany.com"},
                {"url": "https://www.linkedin.com/company/testcompany"}
            ]
        }
        mock_post.return_value = mock_response

        state = {"company_name": "TestCompany"}
        result_state = search_discovery(state)

        self.assertIn("search_urls", result_state)
        self.assertEqual(len(result_state["search_urls"]), 2)
        self.assertEqual(result_state["search_urls"][0], "https://www.testcompany.com")

    @patch('corporate_intelligence_agent.requests.get')
    def test_deep_scraper(self, mock_get):
        # Mock the website request response
        mock_response = MagicMock()
        mock_response.text = "<html><body><p>Test Company is great.</p></body></html>"
        mock_get.return_value = mock_response

        state = {
            "company_name": "TestCompany",
            "search_urls": [
                "https://www.testcompany.com",
                "https://www.linkedin.com/company/testcompany" # Should be skipped
            ]
        }

        result_state = deep_scraper(state)

        self.assertIn("scraped_data", result_state)
        # It should skip linkedin, so it only scrapes testcompany.com
        self.assertIn("Test Company is great", result_state["scraped_data"])
        # Only 1 URL should have been called (linkedin skipped)
        mock_get.assert_called_once()

    @patch('corporate_intelligence_agent.requests.post')
    def test_data_extraction_synthesis(self, mock_post):
        # Mock the OpenAI API response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "# Corporate Intelligence Report: TestCompany\n\n## Executive Summary..."
                    }
                }
            ]
        }
        mock_post.return_value = mock_response

        state = {
            "company_name": "TestCompany",
            "scraped_data": "Test Company is great. Source: testcompany.com"
        }

        result_state = data_extraction_synthesis(state)

        self.assertIn("final_report", result_state)
        self.assertTrue(result_state["final_report"].startswith("# Corporate Intelligence Report: TestCompany"))

    @patch('corporate_intelligence_agent.requests.get')
    @patch('corporate_intelligence_agent.requests.post')
    def test_full_pipeline(self, mock_post, mock_get):
        # Mock Tavily response (First post call)
        mock_tavily_response = MagicMock()
        mock_tavily_response.json.return_value = {
            "results": [{"url": "https://www.testcompany.com"}]
        }

        # Mock OpenAI response (Second post call)
        mock_openai_response = MagicMock()
        mock_openai_response.json.return_value = {
            "choices": [{"message": {"content": "Final Report Context"}}]
        }

        # Configure side_effect for post to handle Tavily vs OpenAI
        def mock_post_side_effect(*args, **kwargs):
            if "api.tavily.com" in args[0]:
                return mock_tavily_response
            elif "api.openai.com" in args[0]:
                return mock_openai_response
            return MagicMock()

        mock_post.side_effect = mock_post_side_effect

        # Mock requests.get for website scraping
        mock_get_response = MagicMock()
        mock_get_response.text = "<html><body>Dummy Data</body></html>"
        mock_get.return_value = mock_get_response

        initial_state = {
            "company_name": "TestCompany",
            "search_urls": [],
            "scraped_data": "",
            "final_report": ""
        }

        final_state = app.invoke(initial_state)

        self.assertIn("final_report", final_state)
        self.assertEqual(final_state["final_report"], "Final Report Context")

if __name__ == '__main__':
    unittest.main()
