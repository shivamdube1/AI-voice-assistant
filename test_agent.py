import unittest
from unittest.mock import patch, MagicMock
from corporate_intelligence_agent import search_discovery, deep_scraper, data_extraction_synthesis

class TestAgent(unittest.TestCase):
    def setUp(self):
        self.initial_state = {
            "company_name": "TestCorp",
            "search_urls": [],
            "scraped_data": "",
            "final_report": ""
        }

    @patch('corporate_intelligence_agent.os.environ.get')
    @patch('corporate_intelligence_agent.requests.post')
    def test_search_discovery(self, mock_post, mock_env_get):
        mock_env_get.return_value = 'dummy_tavily_key'
        mock_response = MagicMock()
        mock_response.json.return_value = {'results': [{'url': 'https://testcorp.com'}, {'url': 'https://linkedin.com/testcorp'}]}
        mock_post.return_value = mock_response

        new_state = search_discovery(self.initial_state)
        self.assertEqual(new_state['search_urls'], ['https://testcorp.com', 'https://linkedin.com/testcorp'])

    @patch('corporate_intelligence_agent.requests.get')
    def test_deep_scraper(self, mock_get):
        state = {"search_urls": ['https://testcorp.com']}
        mock_response = MagicMock()
        mock_response.text = '<html><body><h1>Welcome to TestCorp</h1><p>We are a testing company.</p></body></html>'
        mock_get.return_value = mock_response

        new_state = deep_scraper(state)
        self.assertIn('Welcome to TestCorp', new_state['scraped_data'])
        self.assertIn('We are a testing company.', new_state['scraped_data'])

    @patch('corporate_intelligence_agent.os.environ.get')
    @patch('corporate_intelligence_agent.requests.post')
    def test_data_extraction_synthesis(self, mock_post, mock_env_get):
        mock_env_get.return_value = 'dummy_openai_key'
        state = {"company_name": "TestCorp", "scraped_data": "Welcome to TestCorp"}
        mock_response = MagicMock()
        mock_response.json.return_value = {'choices': [{'message': {'content': '# TestCorp Report'}}]}
        mock_post.return_value = mock_response

        new_state = data_extraction_synthesis(state)
        self.assertEqual(new_state['final_report'], '# TestCorp Report')
