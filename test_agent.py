import unittest
from unittest.mock import patch, MagicMock
from corporate_intelligence_agent import app

class TestCorporateIntelligenceAgent(unittest.TestCase):
    def setUp(self):
        self.company_name = "TestCompany Inc."

        # Mock responses
        self.mock_tavily_response = {
            "results": [
                {"url": "https://www.testcompany.com"},
                {"url": "https://www.linkedin.com/company/testcompany"}
            ]
        }

        self.mock_html_content = "<html><body><p>TestCompany Inc. is a leading provider of test services.</p></body></html>"

        self.mock_openai_response_success = {
            "choices": [
                {
                    "message": {
                        "content": "# TestCompany Inc. Report\n\n## Executive Summary\nTestCompany Inc. is a leading provider of test services.\n\n## Company Profile\n- Specialization: Test services.\n- Company Size: 500 employees.\n- Headquarters / Key Locations: Test City.\n\n## Company Performance\n- Financials / Growth: $10M revenue.\n- Market Position: Leader in test services.\n- Core Products & Offerings: Test Tool v1.\n- Recent Developments: Launched new test product."
                    }
                }
            ]
        }

        self.mock_openai_response_missing_data = {
            "choices": [
                {
                    "message": {
                        "content": "# TestCompany Inc. Report\n\n## Executive Summary\nTestCompany Inc. is a leading provider of test services.\n\n## Company Profile\n- Specialization: Test services.\n- Company Size: Insufficient data found for this metric\n- Headquarters / Key Locations: Test City.\n\n## Company Performance\n- Financials / Growth: Insufficient data found for this metric\n- Market Position: Leader in test services.\n- Core Products & Offerings: Test Tool v1.\n- Recent Developments: Launched new test product."
                    }
                }
            ]
        }


    @patch('corporate_intelligence_agent.requests.post')
    @patch('corporate_intelligence_agent.requests.get')
    def test_pipeline_success(self, mock_get, mock_post):
        # Setup mock for GET requests (BeautifulSoup)
        mock_get_response = MagicMock()
        mock_get_response.text = self.mock_html_content
        mock_get_response.raise_for_status.return_value = None
        mock_get.return_value = mock_get_response

        # Setup mock for POST requests (Tavily and OpenAI)
        def post_side_effect(url, **kwargs):
            mock_response = MagicMock()
            mock_response.raise_for_status.return_value = None
            if "tavily" in url:
                mock_response.json.return_value = self.mock_tavily_response
            elif "openai" in url:
                mock_response.json.return_value = self.mock_openai_response_success
            return mock_response

        mock_post.side_effect = post_side_effect

        initial_state = {"company_name": self.company_name}
        final_state = app.invoke(initial_state)

        report = final_state.get("report", "")

        # Check that the required sections are present
        self.assertIn("Executive Summary", report)
        self.assertIn("Company Profile", report)
        self.assertIn("Specialization:", report)
        self.assertIn("Company Size:", report)
        self.assertIn("Headquarters / Key Locations:", report)
        self.assertIn("Company Performance", report)
        self.assertIn("Financials / Growth:", report)
        self.assertIn("Market Position:", report)
        self.assertIn("Core Products & Offerings:", report)
        self.assertIn("Recent Developments:", report)

    @patch('corporate_intelligence_agent.requests.post')
    @patch('corporate_intelligence_agent.requests.get')
    def test_pipeline_missing_data(self, mock_get, mock_post):
        # Setup mock for GET requests
        mock_get_response = MagicMock()
        mock_get_response.text = self.mock_html_content
        mock_get_response.raise_for_status.return_value = None
        mock_get.return_value = mock_get_response

        # Setup mock for POST requests with missing data response from OpenAI
        def post_side_effect(url, **kwargs):
            mock_response = MagicMock()
            mock_response.raise_for_status.return_value = None
            if "tavily" in url:
                mock_response.json.return_value = self.mock_tavily_response
            elif "openai" in url:
                mock_response.json.return_value = self.mock_openai_response_missing_data
            return mock_response

        mock_post.side_effect = post_side_effect

        initial_state = {"company_name": self.company_name}
        final_state = app.invoke(initial_state)

        report = final_state.get("report", "")

        # Check that the constraint is followed
        self.assertIn("Insufficient data found for this metric", report)

if __name__ == '__main__':
    unittest.main()
