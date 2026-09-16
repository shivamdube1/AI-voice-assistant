import unittest
from unittest.mock import patch, MagicMock
import os
from corporate_intelligence_agent import build_graph

class TestCorporateIntelligenceAgent(unittest.TestCase):
    def setUp(self):
        # Set dummy keys so fallback is NOT triggered, allowing API code paths to run
        os.environ["TAVILY_API_KEY"] = "dummy_tavily"
        os.environ["OPENAI_API_KEY"] = "dummy_openai"

    def tearDown(self):
        # Clean up env
        if "TAVILY_API_KEY" in os.environ:
            del os.environ["TAVILY_API_KEY"]
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]

    @patch('corporate_intelligence_agent.requests.post')
    @patch('corporate_intelligence_agent.requests.get')
    def test_full_pipeline(self, mock_get, mock_post):
        # Setup mock for Tavily (search_discovery node - post)
        tavily_response = MagicMock()
        tavily_response.json.return_value = {
            "results": [
                {"url": "https://www.testco.com"},
                {"url": "https://www.linkedin.com/company/testco"}
            ]
        }

        # Setup mock for OpenAI (data_extraction_synthesis node - post)
        openai_response = MagicMock()
        openai_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "# Executive Summary\nTestCo is a test company.\n\n# Company Profile\nSpecialization: Testing\nCompany Size: 50\nHeadquarters / Key Locations: Test City\n\n# Company Performance\nFinancials / Growth: Insufficient data found for this metric.\nMarket Position: Other testers\nCore Products & Offerings: Test tools\nRecent Developments: Launched new test"
                    }
                }
            ]
        }

        # mock_post will be called twice (Tavily, then OpenAI)
        mock_post.side_effect = [tavily_response, openai_response]

        # Setup mock for Deep Scraper (deep_scraper node - get)
        scraper_response = MagicMock()
        scraper_response.status_code = 200
        scraper_response.text = "<html><body>We are TestCo, we do testing with 50 employees.</body></html>"
        mock_get.return_value = scraper_response

        # Build graph and run
        app = build_graph()
        initial_state = {"company_name": "TestCo", "search_urls": [], "scraped_data": "", "final_report": ""}

        final_state = None
        for state_update in app.stream(initial_state):
            if "data_extraction_synthesis" in state_update:
                final_state = state_update["data_extraction_synthesis"]
            elif "deep_scraper" in state_update:
                final_state = state_update["deep_scraper"]
            elif "search_discovery" in state_update:
                final_state = state_update["search_discovery"]

        # Assertions
        self.assertIsNotNone(final_state)
        # Check URLs from Tavily mock
        self.assertEqual(final_state["search_urls"], ["https://www.testco.com", "https://www.linkedin.com/company/testco"])
        # Check scraped data (ensure linkedin was skipped and content scraped)
        self.assertIn("Source: https://www.testco.com", final_state["scraped_data"])
        self.assertIn("We are TestCo", final_state["scraped_data"])
        self.assertNotIn("linkedin.com", final_state["scraped_data"]) # shouldn't scrape linkedin content
        # Check final report from OpenAI mock
        self.assertIn("TestCo is a test company.", final_state["final_report"])

if __name__ == '__main__':
    unittest.main()
