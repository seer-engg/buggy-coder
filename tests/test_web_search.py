import unittest
from src.web_search import perform_web_search

class TestWebSearch(unittest.TestCase):
    def test_perform_web_search(self):
        # Test the web search function with a sample query
        query = "OpenAI"
        results = perform_web_search(query)
        
        # Check if results are returned
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        
        # Check if each result is a tuple with title and link
        for result in results:
            self.assertIsInstance(result, tuple)
            self.assertEqual(len(result), 2)
            self.assertIsInstance(result[0], str)  # Title
            self.assertIsInstance(result[1], str)  # Link

if __name__ == "__main__":
    unittest.main()