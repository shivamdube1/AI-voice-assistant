import unittest
from unittest.mock import patch, MagicMock
from corporate_intelligence_agent import (
    search_discovery,
    deep_scraper,
    data_extraction_synthesis,
    app,
    AgentState
)

class TestCorporateIntelligenceAgent(unittest.TestCase):

    @patch('corporate_intelligence_agent.requests.post')
    @patch('corporate_intelligence_agent.os.environ.get')
    def test_search_discovery(self, mock_env_get, mock_post):
        mock_env_get.return_value = "fake_tavily_key"

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"url": "https://example.com/1"},
                {"url": "https://example.com/2"}
            ]
        }
        mock_post.return_value = mock_response

        initial_state: AgentState = {
            "company_name": "TestCorp",
            "search_urls": [],
            "scraped_data": "",
            "final_report": ""
        }

        new_state = search_discovery(initial_state)

        self.assertEqual(new_state["company_name"], "TestCorp")
        self.assertEqual(new_state["search_urls"], ["https://example.com/1", "https://example.com/2"])
        self.assertEqual(new_state["scraped_data"], "")
        self.assertEqual(new_state["final_report"], "")

    @patch('corporate_intelligence_agent.requests.get')
    def test_deep_scraper(self, mock_get):
        mock_response = MagicMock()
        mock_response.text = "<html><body><p>This is test content.</p></body></html>"
        mock_get.return_value = mock_response

        initial_state: AgentState = {
            "company_name": "TestCorp",
            "search_urls": ["https://example.com/1"],
            "scraped_data": "",
            "final_report": ""
        }

        new_state = deep_scraper(initial_state)

        self.assertEqual(new_state["company_name"], "TestCorp")
        self.assertEqual(new_state["search_urls"], ["https://example.com/1"])
        self.assertIn("This is test content.", new_state["scraped_data"])
        self.assertIn("Source: https://example.com/1", new_state["scraped_data"])

    @patch('corporate_intelligence_agent.requests.post')
    @patch('corporate_intelligence_agent.os.environ.get')
    def test_data_extraction_synthesis(self, mock_env_get, mock_post):
        mock_env_get.return_value = "fake_openai_key"

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "This is the final report for TestCorp."
                    }
                }
            ]
        }
        mock_post.return_value = mock_response

        initial_state: AgentState = {
            "company_name": "TestCorp",
            "search_urls": ["https://example.com/1"],
            "scraped_data": "This is test content.",
            "final_report": ""
        }

        new_state = data_extraction_synthesis(initial_state)

        self.assertEqual(new_state["company_name"], "TestCorp")
        self.assertEqual(new_state["search_urls"], ["https://example.com/1"])
        self.assertEqual(new_state["scraped_data"], "This is test content.")
        self.assertEqual(new_state["final_report"], "This is the final report for TestCorp.")

    @patch('corporate_intelligence_agent.requests.post')
    @patch('corporate_intelligence_agent.requests.get')
    @patch('corporate_intelligence_agent.os.environ.get')
    def test_full_pipeline(self, mock_env_get, mock_get, mock_post):
        # Mock env vars
        def mock_env_get_side_effect(key, default=""):
            if key == "TAVILY_API_KEY":
                return "fake_tavily_key"
            if key == "OPENAI_API_KEY":
                return "fake_openai_key"
            return default
        mock_env_get.side_effect = mock_env_get_side_effect

        # Mock requests.post (used for both Tavily and OpenAI)
        def mock_post_side_effect(url, *args, **kwargs):
            mock_response = MagicMock()
            if "tavily" in url:
                mock_response.json.return_value = {
                    "results": [{"url": "https://example.com/test"}]
                }
            elif "openai" in url:
                mock_response.json.return_value = {
                    "choices": [{"message": {"content": "Final Report Document."}}]
                }
            return mock_response
        mock_post.side_effect = mock_post_side_effect

        # Mock requests.get (used for BeautifulSoup scraping)
        mock_get_response = MagicMock()
        mock_get_response.text = "<p>Company info here.</p>"
        mock_get.return_value = mock_get_response

        initial_state: AgentState = {
            "company_name": "PipelineCorp",
            "search_urls": [],
            "scraped_data": "",
            "final_report": ""
        }

        result_state = app.invoke(initial_state)

        self.assertEqual(result_state["company_name"], "PipelineCorp")
        self.assertEqual(result_state["search_urls"], ["https://example.com/test"])
        self.assertIn("Company info here.", result_state["scraped_data"])
        self.assertEqual(result_state["final_report"], "Final Report Document.")

if __name__ == '__main__':
    unittest.main()
