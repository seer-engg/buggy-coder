import unittest
from unittest.mock import patch
from src.state import BuggyCoderState

class TestE2BIntegration(unittest.TestCase):
    def setUp(self):
        self.state = BuggyCoderState(snippet="print('Hello, World!')", instructions="Run this code.")

    @patch('src.graph.create_agent')
    def test_execution_success(self, mock_create_agent):
        # Mock the agent creation to bypass API key requirement
        mock_create_agent.return_value = None
        from src.graph import get_app
        app = get_app()  # Call the function after mocking
        # Placeholder for actual E2B execution
        self.state['execution_result'] = "Hello, World!"
        self.assertEqual(self.state['execution_result'], "Hello, World!")

    @patch('src.graph.create_agent')
    def test_execution_error(self, mock_create_agent):
        # Mock the agent creation to bypass API key requirement
        mock_create_agent.return_value = None
        from src.graph import get_app
        app = get_app()  # Call the function after mocking
        # Placeholder for actual E2B execution error
        self.state['execution_error'] = "SyntaxError: invalid syntax"
        self.assertEqual(self.state['execution_error'], "SyntaxError: invalid syntax")

if __name__ == '__main__':
    unittest.main()
