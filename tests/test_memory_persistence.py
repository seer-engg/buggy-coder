import unittest
from src.state import PersistentMemory

class TestPersistentMemory(unittest.TestCase):
    def setUp(self):
        """Set up a temporary database for testing."""
        self.memory = PersistentMemory(db_path=':memory:')

    def tearDown(self):
        """Close the database connection after each test."""
        self.memory.close()

    def test_store_and_retrieve_message(self):
        """Test storing and retrieving a single message."""
        test_message = "Hello, world!"
        self.memory.store_message(test_message)
        retrieved_messages = self.memory.retrieve_messages()
        self.assertIn(test_message, retrieved_messages)

    def test_retrieve_multiple_messages(self):
        """Test storing and retrieving multiple messages."""
        messages = ["Hello, world!", "How are you?", "Goodbye!"]
        for message in messages:
            self.memory.store_message(message)
        retrieved_messages = self.memory.retrieve_messages()
        self.assertEqual(messages, retrieved_messages)

if __name__ == '__main__':
    unittest.main()
