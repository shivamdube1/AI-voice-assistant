import unittest
from unittest.mock import patch, MagicMock
from corporate_intelligence_agent import (
    search_discovery,
    deep_scraper,
    data_extraction_synthesis,
    build_graph,
    AgentState
)

class TestCorporateIntelligenceAgent(unittest.TestCase):

    @patch('corporate_intelligence_agent.os.environ.get')
    @patch('corporate_intelligence_agent.requests.post')
    def test_search_discovery(self, mock_post, mock_getenv):
        # Mock API key presence
        mock_getenv.return_value = "fake_tavily_key"

        # Mock Tavily response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"url": "https://www.example.com"},
                {"url": "https://www.linkedin.com/company/example"}
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        state: AgentState = {"company_name": "Example Corp"}
        new_state = search_discovery(state)

        self.assertEqual(len(new_state["search_urls"]), 2)
        self.assertIn("https://www.example.com", new_state["search_urls"])
        mock_post.assert_called_once()

    @patch('corporate_intelligence_agent.requests.get')
    def test_deep_scraper(self, mock_get):
        # Mock requests.get response
        mock_response = MagicMock()
        mock_response.text = "<html><body><script>var x = 1;</script><h1>Example Corp</h1><p>We have 500 employees.</p></body></html>"
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        state: AgentState = {"company_name": "Example Corp", "search_urls": ["https://www.example.com"]}
        new_state = deep_scraper(state)

        scraped = new_state.get("scraped_data", "")
        self.assertIn("Example Corp We have 500 employees.", scraped)
        self.assertNotIn("var x = 1;", scraped)
        mock_get.assert_called_once()

    @patch('corporate_intelligence_agent.os.environ.get')
    @patch('corporate_intelligence_agent.requests.post')
    def test_data_extraction_synthesis(self, mock_post, mock_getenv):
        # Mock API key presence
        mock_getenv.return_value = "fake_openai_key"

        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {"message": {"content": "# Executive Summary\nExample Corp is a test company."}}
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        state: AgentState = {"company_name": "Example Corp", "scraped_data": "Example Corp We have 500 employees."}
        new_state = data_extraction_synthesis(state)

        self.assertEqual(new_state.get("final_report"), "# Executive Summary\nExample Corp is a test company.")
        mock_post.assert_called_once()

    def test_build_graph(self):
        graph = build_graph()
        self.assertIsNotNone(graph)

if __name__ == '__main__':
    unittest.main()
