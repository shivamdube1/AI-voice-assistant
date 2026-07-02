import unittest
from unittest.mock import patch, MagicMock
from corporate_intelligence_agent import search_discovery, deep_scraper, data_extraction_synthesis, graph

class TestCorporateIntelligenceAgent(unittest.TestCase):
    @patch('corporate_intelligence_agent.os.environ.get')
    @patch('corporate_intelligence_agent.requests.post')
    def test_search_discovery(self, mock_post, mock_getenv):
        # Setup mock environment variable
        mock_getenv.return_value = 'fake_tavily_api_key'

        # Setup mock response for requests.post
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {"url": "https://example.com"},
                {"url": "https://example.com/linkedin"}
            ]
        }
        mock_post.return_value = mock_response

        state = {"company_name": "Example Corp"}
        new_state = search_discovery(state)

        self.assertIn("search_urls", new_state)
        self.assertEqual(len(new_state["search_urls"]), 2)
        self.assertEqual(new_state["search_urls"][0], "https://example.com")

    @patch('corporate_intelligence_agent.requests.get')
    def test_deep_scraper(self, mock_get):
        # Setup mock response for requests.get
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"<html><body><p>We are a tech company with 500 employees.</p></body></html>"
        mock_get.return_value = mock_response

        state = {"search_urls": ["https://example.com"]}
        new_state = deep_scraper(state)

        self.assertIn("scraped_data", new_state)
        self.assertIn("tech company", new_state["scraped_data"])
        self.assertIn("500 employees", new_state["scraped_data"])

    @patch('corporate_intelligence_agent.os.environ.get')
    @patch('corporate_intelligence_agent.requests.post')
    def test_data_extraction_synthesis(self, mock_post, mock_getenv):
        # Setup mock environment variable
        mock_getenv.return_value = 'fake_openai_api_key'

        # Setup mock response for requests.post
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "### Executive Summary\nExample Corp is a tech company."
                    }
                }
            ]
        }
        mock_post.return_value = mock_response

        state = {
            "company_name": "Example Corp",
            "scraped_data": "Source: https://example.com\nWe are a tech company with 500 employees."
        }
        new_state = data_extraction_synthesis(state)

        self.assertIn("final_report", new_state)
        self.assertIn("Executive Summary", new_state["final_report"])

    @patch('corporate_intelligence_agent.os.environ.get')
    @patch('corporate_intelligence_agent.requests.post')
    @patch('corporate_intelligence_agent.requests.get')
    def test_full_graph_invocation(self, mock_get, mock_post, mock_getenv):
        def side_effect_post(*args, **kwargs):
            mock_response = MagicMock()
            mock_response.status_code = 200
            if "tavily" in args[0]:
                mock_response.json.return_value = {
                    "results": [{"url": "https://example.com"}]
                }
            elif "openai" in args[0]:
                mock_response.json.return_value = {
                    "choices": [
                        {"message": {"content": "Final Example Report"}}
                    ]
                }
            return mock_response

        mock_post.side_effect = side_effect_post

        mock_response_get = MagicMock()
        mock_response_get.status_code = 200
        mock_response_get.content = b"<html><body>Data</body></html>"
        mock_get.return_value = mock_response_get

        mock_getenv.return_value = 'fake_api_key'

        initial_state = {"company_name": "Example Corp"}
        final_state = graph.invoke(initial_state)

        self.assertIn("search_urls", final_state)
        self.assertIn("scraped_data", final_state)
        self.assertIn("final_report", final_state)
        self.assertEqual(final_state["final_report"], "Final Example Report")

if __name__ == '__main__':
    unittest.main()
