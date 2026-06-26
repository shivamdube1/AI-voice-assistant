import unittest
from unittest.mock import patch
from corporate_intelligence_agent import search_discovery, deep_scraper, data_extraction_synthesis

class TestCorporateIntelligenceAgent(unittest.TestCase):

    @patch('corporate_intelligence_agent.requests.post')
    def test_search_discovery(self, mock_post):
        mock_response = unittest.mock.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {"url": "https://example.com", "content": "Example Content"}
            ]
        }
        mock_post.return_value = mock_response

        state = {"company_name": "Example Corp"}
        result_state = search_discovery(state)

        self.assertIn("search_results", result_state)
        self.assertEqual(len(result_state["search_results"]), 1)
        self.assertEqual(result_state["search_results"][0]["url"], "https://example.com")
        mock_post.assert_called_once()

    @patch('corporate_intelligence_agent.requests.get')
    def test_deep_scraper(self, mock_get):
        mock_response = unittest.mock.Mock()
        mock_response.status_code = 200
        mock_response.text = "<html><body><h1>Example Company</h1><p>Some text.</p></body></html>"
        mock_get.return_value = mock_response

        state = {
            "search_results": [{"url": "https://example.com"}]
        }
        result_state = deep_scraper(state)

        self.assertIn("scraped_data", result_state)
        self.assertIn("Example Company", result_state["scraped_data"])
        self.assertIn("Some text.", result_state["scraped_data"])
        mock_get.assert_called_once()

    @patch('corporate_intelligence_agent.requests.post')
    def test_data_extraction_synthesis(self, mock_post):
        mock_response = unittest.mock.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "# Report\n\nExecutive Summary..."
                    }
                }
            ]
        }
        mock_post.return_value = mock_response

        state = {
            "company_name": "Example Corp",
            "scraped_data": "Some scraped data."
        }
        result_state = data_extraction_synthesis(state)

        self.assertIn("report", result_state)
        self.assertEqual(result_state["report"], "# Report\n\nExecutive Summary...")
        mock_post.assert_called_once()

if __name__ == '__main__':
    unittest.main()
