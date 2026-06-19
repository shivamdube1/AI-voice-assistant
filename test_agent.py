import unittest
from unittest.mock import patch, MagicMock
from corporate_intelligence_agent import (
    search_discovery,
    deep_scraper,
    data_extraction_synthesis,
    app
)

class TestCorporateIntelligenceAgent(unittest.TestCase):

    @patch('corporate_intelligence_agent.os.environ.get')
    @patch('corporate_intelligence_agent.requests.post')
    def test_search_discovery(self, mock_post, mock_env_get):
        # Mock environment variable
        mock_env_get.return_value = "dummy_tavily_key"

        # Mock Tavily API response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"url": "https://www.example.com"},
                {"url": "https://www.example.com/about"}
            ]
        }
        mock_post.return_value = mock_response

        initial_state = {"company_name": "TestCorp"}
        result = search_discovery(initial_state)

        # Check that we made 3 calls (one for each query)
        self.assertEqual(mock_post.call_count, 3)

        urls = result.get("urls_to_scrape", [])
        self.assertIn("https://www.example.com", urls)
        self.assertIn("https://www.example.com/about", urls)
        self.assertEqual(len(urls), 2)  # because set() removes duplicates

    @patch('corporate_intelligence_agent.requests.get')
    def test_deep_scraper(self, mock_get):
        # Mock website response
        mock_response = MagicMock()
        mock_response.text = "<html><body><h1>TestCorp</h1><p>We are a test company.</p></body></html>"
        mock_get.return_value = mock_response

        state = {"urls_to_scrape": ["https://www.example.com"]}
        result = deep_scraper(state)

        scraped_data = result.get("scraped_data", "")
        self.assertIn("TestCorp", scraped_data)
        self.assertIn("We are a test company.", scraped_data)
        self.assertIn("Source: https://www.example.com", scraped_data)

    @patch('corporate_intelligence_agent.os.environ.get')
    @patch('corporate_intelligence_agent.requests.post')
    def test_data_extraction_synthesis(self, mock_post, mock_env_get):
        # Mock environment variable
        mock_env_get.return_value = "dummy_openai_key"

        # Mock OpenAI API response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "# TestCorp Report\n\nExecutive Summary..."
                    }
                }
            ]
        }
        mock_post.return_value = mock_response

        state = {"scraped_data": "Some raw scraped data...", "company_name": "TestCorp"}
        result = data_extraction_synthesis(state)

        self.assertEqual(mock_post.call_count, 1)
        self.assertEqual(result.get("report"), "# TestCorp Report\n\nExecutive Summary...")

    @patch('corporate_intelligence_agent.os.environ.get')
    @patch('corporate_intelligence_agent.requests.get')
    @patch('corporate_intelligence_agent.requests.post')
    def test_graph_execution(self, mock_post, mock_get, mock_env_get):
        # Mock API Keys (accepting multiple arguments including defaults)
        def env_side_effect(key, default=None):
            if key == "TAVILY_API_KEY":
                return "dummy_tavily_key"
            elif key == "OPENAI_API_KEY":
                return "dummy_openai_key"
            return default
        mock_env_get.side_effect = env_side_effect

        # Configure mock_post for both Tavily and OpenAI
        def post_side_effect(url, **kwargs):
            mock_response = MagicMock()
            if "tavily" in url:
                mock_response.json.return_value = {
                    "results": [{"url": "https://www.testcorp.com"}]
                }
            elif "openai" in url:
                mock_response.json.return_value = {
                    "choices": [
                        {"message": {"content": "Final Report Generation"}}
                    ]
                }
            return mock_response

        mock_post.side_effect = post_side_effect

        # Configure mock_get for scraping
        mock_get_response = MagicMock()
        mock_get_response.text = "<html><body>Some scraped data</body></html>"
        mock_get.return_value = mock_get_response

        initial_state = {
            "company_name": "TestCorp",
            "search_queries": [],
            "urls_to_scrape": [],
            "scraped_data": "",
            "report": ""
        }

        result = app.invoke(initial_state)

        self.assertEqual(result.get("company_name"), "TestCorp")
        self.assertIn("https://www.testcorp.com", result.get("urls_to_scrape", []))
        self.assertIn("Some scraped data", result.get("scraped_data", ""))
        self.assertEqual(result.get("report"), "Final Report Generation")

if __name__ == '__main__':
    unittest.main()
