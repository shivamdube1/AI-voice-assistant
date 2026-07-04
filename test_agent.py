import unittest
from unittest.mock import patch, MagicMock
from corporate_intelligence_agent import search_discovery, deep_scraper, data_extraction_synthesis, build_graph

class TestCorporateIntelligenceAgent(unittest.TestCase):

    @patch('corporate_intelligence_agent.os.environ.get')
    @patch('corporate_intelligence_agent.requests.post')
    def test_search_discovery(self, mock_post, mock_getenv):
        # Mock API key presence
        mock_getenv.return_value = 'mock_tavily_api_key'

        # Mock requests.post response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {"url": "https://www.example.com"},
                {"url": "https://www.linkedin.com/company/example"}
            ]
        }
        mock_post.return_value = mock_response

        state = {"company_name": "Example Corp", "search_urls": [], "scraped_data": "", "final_report": ""}
        new_state = search_discovery(state)

        self.assertEqual(len(new_state["search_urls"]), 2)
        self.assertIn("https://www.example.com", new_state["search_urls"])

    @patch('corporate_intelligence_agent.requests.get')
    def test_deep_scraper(self, mock_get):
        # Mock requests.get response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html><body><p>We have 500 employees. Our revenue is $10M.</p></body></html>"
        mock_get.return_value = mock_response

        state = {"company_name": "Example Corp", "search_urls": ["https://www.example.com"], "scraped_data": "", "final_report": ""}
        new_state = deep_scraper(state)

        self.assertIn("500 employees", new_state["scraped_data"])
        self.assertIn("$10M", new_state["scraped_data"])

    @patch('corporate_intelligence_agent.os.environ.get')
    @patch('corporate_intelligence_agent.requests.post')
    def test_data_extraction_synthesis(self, mock_post, mock_getenv):
        # Mock API key presence
        mock_getenv.return_value = 'mock_openai_api_key'

        # Mock requests.post response for OpenAI
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "# Executive Summary\nExample Corp is a great company.\n# Company Size\n500 employees."
                    }
                }
            ]
        }
        mock_post.return_value = mock_response

        state = {"company_name": "Example Corp", "search_urls": [], "scraped_data": "We have 500 employees.", "final_report": ""}
        new_state = data_extraction_synthesis(state)

        self.assertIn("500 employees", new_state["final_report"])

    @patch('corporate_intelligence_agent.os.environ.get')
    @patch('corporate_intelligence_agent.requests.post')
    @patch('corporate_intelligence_agent.requests.get')
    def test_graph_pipeline(self, mock_get, mock_post, mock_getenv):
        # Mock environment variables
        def mock_env(key, default=None):
            if key == "TAVILY_API_KEY": return "mock_tavily_api_key"
            if key == "OPENAI_API_KEY": return "mock_openai_api_key"
            return default
        mock_getenv.side_effect = mock_env

        # Mock requests.post (used by both Tavily and OpenAI)
        def mock_post_requests(*args, **kwargs):
            mock_response = MagicMock()
            mock_response.status_code = 200

            if "tavily" in args[0]:
                mock_response.json.return_value = {
                    "results": [{"url": "https://www.example.com"}]
                }
            elif "openai" in args[0]:
                mock_response.json.return_value = {
                    "choices": [{"message": {"content": "# Final Mock Report\nData extracted."}}]
                }
            return mock_response

        mock_post.side_effect = mock_post_requests

        # Mock requests.get (used by BeautifulSoup)
        mock_get_response = MagicMock()
        mock_get_response.status_code = 200
        mock_get_response.text = "<html><body><p>Example Data</p></body></html>"
        mock_get.return_value = mock_get_response

        app = build_graph()
        initial_state = {"company_name": "Test Company", "search_urls": [], "scraped_data": "", "final_report": ""}
        result = app.invoke(initial_state)

        self.assertEqual(len(result["search_urls"]), 1)
        self.assertIn("Example Data", result["scraped_data"])
        self.assertIn("Final Mock Report", result["final_report"])

if __name__ == '__main__':
    unittest.main()
