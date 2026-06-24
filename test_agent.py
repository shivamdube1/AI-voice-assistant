import unittest
from unittest.mock import patch, MagicMock
from corporate_intelligence_agent import build_graph, search_discovery, deep_scraper, data_extraction_synthesis

class TestCorporateIntelligenceAgent(unittest.TestCase):

    @patch('corporate_intelligence_agent.requests.post')
    def test_search_discovery(self, mock_post):
        # Mock Tavily response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {"url": "https://www.example.com"},
                {"url": "https://www.linkedin.com/company/example"}
            ]
        }
        mock_post.return_value = mock_response

        with patch.dict('os.environ', {'TAVILY_API_KEY': 'fake_key'}):
            state = {"company_name": "Example Corp", "search_results": [], "scraped_data": "", "report": ""}
            result_state = search_discovery(state)

            self.assertEqual(result_state["search_results"], ["https://www.example.com", "https://www.linkedin.com/company/example"])
            mock_post.assert_called_once()

    @patch('corporate_intelligence_agent.requests.get')
    def test_deep_scraper(self, mock_get):
        # Mock requests.get to return fake HTML
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"<html><body><p>Example Corp has 500 employees.</p></body></html>"
        mock_get.return_value = mock_response

        state = {
            "company_name": "Example Corp",
            "search_results": ["https://www.example.com"],
            "scraped_data": "",
            "report": ""
        }

        result_state = deep_scraper(state)

        self.assertIn("Example Corp has 500 employees.", result_state["scraped_data"])
        mock_get.assert_called_once()

    @patch('corporate_intelligence_agent.requests.post')
    def test_data_extraction_synthesis(self, mock_post):
        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "# Executive Summary\n\nExample Corp is a great company."
                    }
                }
            ]
        }
        mock_post.return_value = mock_response

        with patch.dict('os.environ', {'OPENAI_API_KEY': 'fake_key'}):
            state = {
                "company_name": "Example Corp",
                "search_results": [],
                "scraped_data": "Example Corp has 500 employees.",
                "report": ""
            }

            result_state = data_extraction_synthesis(state)

            self.assertEqual(result_state["report"], "# Executive Summary\n\nExample Corp is a great company.")
            mock_post.assert_called_once()

    @patch('corporate_intelligence_agent.requests.get')
    @patch('corporate_intelligence_agent.requests.post')
    def test_full_pipeline(self, mock_post, mock_get):
        # Mock Tavily response (First post call)
        tavily_response = MagicMock()
        tavily_response.status_code = 200
        tavily_response.json.return_value = {
            "results": [{"url": "https://www.example.com"}]
        }

        # Mock OpenAI response (Second post call)
        openai_response = MagicMock()
        openai_response.status_code = 200
        openai_response.json.return_value = {
            "choices": [
                {"message": {"content": "Final Report Document"}}
            ]
        }

        # Setup side effect for post to handle different URLs
        def post_side_effect(*args, **kwargs):
            if "tavily.com" in args[0]:
                return tavily_response
            elif "openai.com" in args[0]:
                return openai_response
            return MagicMock(status_code=404)

        mock_post.side_effect = post_side_effect

        # Mock Website Scraping (get call)
        website_response = MagicMock()
        website_response.status_code = 200
        website_response.content = b"<html><p>Test Data</p></html>"
        mock_get.return_value = website_response

        with patch.dict('os.environ', {'TAVILY_API_KEY': 'fake_tavily', 'OPENAI_API_KEY': 'fake_openai'}):
            graph = build_graph()
            initial_state = {"company_name": "Example Corp", "search_results": [], "scraped_data": "", "report": ""}

            final_state = graph.invoke(initial_state)

            self.assertEqual(final_state["report"], "Final Report Document")
            self.assertEqual(mock_post.call_count, 2)
            self.assertEqual(mock_get.call_count, 1)

if __name__ == '__main__':
    unittest.main()
