import unittest
from unittest.mock import patch, MagicMock
import os
from corporate_intelligence_agent import search_discovery, deep_scraper, data_extraction_synthesis, build_graph, AgentState

class TestCorporateIntelligenceAgent(unittest.TestCase):

    @patch.dict(os.environ, {"TAVILY_API_KEY": "test_tavily_key"})
    @patch("corporate_intelligence_agent.requests.post")
    def test_search_discovery(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {"url": "https://example.com", "title": "Example", "content": "Example content"}
            ]
        }
        mock_post.return_value = mock_response

        state: AgentState = {"company_name": "Test Company", "search_results": None, "scraped_data": None, "final_report": None}
        result = search_discovery(state)

        self.assertIn("search_results", result)
        self.assertEqual(len(result["search_results"]), 1)
        self.assertEqual(result["search_results"][0]["url"], "https://example.com")
        mock_post.assert_called_once()

    @patch("corporate_intelligence_agent.requests.get")
    def test_deep_scraper(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"<html><body><p>Test scraped content</p></body></html>"
        mock_get.return_value = mock_response

        state: AgentState = {
            "company_name": "Test Company",
            "search_results": [{"url": "https://example.com", "title": "Example", "content": "Example content"}],
            "scraped_data": None,
            "final_report": None
        }
        result = deep_scraper(state)

        self.assertIn("scraped_data", result)
        self.assertIn("Test scraped content", result["scraped_data"])
        mock_get.assert_called_once_with("https://example.com", timeout=10)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_openai_key"})
    @patch("corporate_intelligence_agent.requests.post")
    def test_data_extraction_synthesis(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {"message": {"content": "# Final Report\n\nExecutive Summary\n\nTest content"}}
            ]
        }
        mock_post.return_value = mock_response

        state: AgentState = {
            "company_name": "Test Company",
            "search_results": None,
            "scraped_data": "Test scraped content",
            "final_report": None
        }
        result = data_extraction_synthesis(state)

        self.assertIn("final_report", result)
        self.assertIn("Test content", result["final_report"])
        mock_post.assert_called_once()

    @patch.dict(os.environ, {"TAVILY_API_KEY": "test_tavily_key", "OPENAI_API_KEY": "test_openai_key"})
    @patch("corporate_intelligence_agent.requests.get")
    @patch("corporate_intelligence_agent.requests.post")
    def test_full_pipeline(self, mock_post, mock_get):
        # Setup mock for Tavily POST
        tavily_response = MagicMock()
        tavily_response.status_code = 200
        tavily_response.json.return_value = {
            "results": [
                {"url": "https://example.com", "title": "Example", "content": "Example content"}
            ]
        }

        # Setup mock for OpenAI POST
        openai_response = MagicMock()
        openai_response.status_code = 200
        openai_response.json.return_value = {
            "choices": [
                {"message": {"content": "# Final Report\n\nExecutive Summary\n\nTest content"}}
            ]
        }

        # mock_post is called twice: once for Tavily, once for OpenAI
        mock_post.side_effect = [tavily_response, openai_response]

        # Setup mock for GET
        mock_get_response = MagicMock()
        mock_get_response.status_code = 200
        mock_get_response.content = b"<html><body><p>Test scraped content</p></body></html>"
        mock_get.return_value = mock_get_response

        app = build_graph()
        initial_state: AgentState = {
            "company_name": "Test Company",
            "search_results": None,
            "scraped_data": None,
            "final_report": None
        }

        final_state = app.invoke(initial_state)

        self.assertIn("final_report", final_state)
        self.assertIn("Test content", final_state["final_report"])

if __name__ == '__main__':
    unittest.main()
