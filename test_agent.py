import unittest
from unittest.mock import patch, MagicMock
from corporate_intelligence_agent import search_discovery, deep_scraper, data_extraction_synthesis, build_graph

class TestCorporateIntelligenceAgent(unittest.TestCase):

    @patch('corporate_intelligence_agent.requests.post')
    @patch.dict('os.environ', {'TAVILY_API_KEY': 'test_tavily_key'})
    def test_search_discovery(self, mock_post):
        # Mock Tavily response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {"url": "https://www.testcompany.com"},
                {"url": "https://www.linkedin.com/company/testcompany"}
            ]
        }
        mock_post.return_value = mock_response

        state = {"company_name": "Test Company", "search_urls": [], "scraped_data": "", "final_report": ""}
        result = search_discovery(state)

        self.assertEqual(len(result["search_urls"]), 2)
        self.assertEqual(result["search_urls"][0], "https://www.testcompany.com")
        self.assertEqual(result["search_urls"][1], "https://www.linkedin.com/company/testcompany")
        mock_post.assert_called_once()

    @patch('corporate_intelligence_agent.requests.get')
    def test_deep_scraper(self, mock_get):
        # Mock web responses
        mock_response_1 = MagicMock()
        mock_response_1.status_code = 200
        mock_response_1.text = "<html><body><h1>Test Company</h1><p>We are a test company with 100 employees.</p></body></html>"

        mock_response_2 = MagicMock()
        mock_response_2.status_code = 200
        mock_response_2.text = "<html><body><h1>LinkedIn</h1><p>Test Company is located in NY.</p></body></html>"

        mock_get.side_effect = [mock_response_1, mock_response_2]

        state = {
            "company_name": "Test Company",
            "search_urls": ["https://www.testcompany.com", "https://www.linkedin.com/company/testcompany"],
            "scraped_data": "",
            "final_report": ""
        }

        result = deep_scraper(state)

        self.assertIn("Test Company", result["scraped_data"])
        self.assertIn("100 employees", result["scraped_data"])
        self.assertIn("located in NY", result["scraped_data"])
        self.assertEqual(mock_get.call_count, 2)

    @patch('corporate_intelligence_agent.requests.post')
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test_openai_key'})
    def test_data_extraction_synthesis(self, mock_post):
        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.status_code = 200
        expected_report = "# Executive Summary\\nTest Company is great."
        mock_response.json.return_value = {
            "choices": [
                {"message": {"content": expected_report}}
            ]
        }
        mock_post.return_value = mock_response

        state = {
            "company_name": "Test Company",
            "search_urls": [],
            "scraped_data": "Raw data about Test Company",
            "final_report": ""
        }

        result = data_extraction_synthesis(state)

        self.assertEqual(result["final_report"], expected_report)
        mock_post.assert_called_once()

    @patch('corporate_intelligence_agent.requests.post')
    @patch('corporate_intelligence_agent.requests.get')
    @patch.dict('os.environ', {'TAVILY_API_KEY': 'test_tavily_key', 'OPENAI_API_KEY': 'test_openai_key'})
    def test_full_graph_execution(self, mock_get, mock_post):
        # Mock Tavily Post (1st call to post)
        tavily_mock_response = MagicMock()
        tavily_mock_response.status_code = 200
        tavily_mock_response.json.return_value = {
            "results": [{"url": "https://www.testcompany.com"}]
        }

        # Mock OpenAI Post (2nd call to post)
        openai_mock_response = MagicMock()
        openai_mock_response.status_code = 200
        openai_mock_response.json.return_value = {
            "choices": [{"message": {"content": "Final Report Mocked"}}]
        }

        mock_post.side_effect = [tavily_mock_response, openai_mock_response]

        # Mock Get
        get_mock_response = MagicMock()
        get_mock_response.status_code = 200
        get_mock_response.text = "<html><body>Data</body></html>"
        mock_get.return_value = get_mock_response

        graph = build_graph()
        initial_state = {
            "company_name": "Test Company",
            "search_urls": [],
            "scraped_data": "",
            "final_report": ""
        }

        result = graph.invoke(initial_state)

        self.assertEqual(result["final_report"], "Final Report Mocked")
        self.assertEqual(mock_post.call_count, 2)
        self.assertEqual(mock_get.call_count, 1)

if __name__ == '__main__':
    unittest.main()
