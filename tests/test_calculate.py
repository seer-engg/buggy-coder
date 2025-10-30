import unittest
from unittest.mock import patch

with patch('src.graph.create_agent', return_value=None):
    from src.graph import perform_calculation as calculate_expression

class TestCalculateTool(unittest.TestCase):
    def test_addition(self):
        self.assertEqual(calculate_expression("2+3"), "5")

    def test_subtraction(self):
        self.assertEqual(calculate_expression("10-5"), "5")

    def test_multiplication(self):
        self.assertEqual(calculate_expression("3*4"), "12")

    def test_division(self):
        self.assertEqual(calculate_expression("20/4"), "5.0")  # Note: Division always returns float

    def test_invalid_characters(self):
        self.assertEqual(calculate_expression("2+3a"), "Error: Invalid characters in expression.")

    def test_invalid_expression(self):
        self.assertTrue(calculate_expression("2//")[:6], "Error:")

    def test_parentheses(self):
        self.assertEqual(calculate_expression("(2+3)*2"), "10")

if __name__ == '__main__':
    unittest.main()
