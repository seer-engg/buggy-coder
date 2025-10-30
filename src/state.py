from typing import Annotated, TypedDict
import sqlite3

from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage

class IOState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]

class BuggyCoderState(IOState):
    snippet: str
    instructions: str

# Persistent memory storage using SQLite
class PersistentMemory:
    def __init__(self, db_path='memory.db'):
        self.connection = sqlite3.connect(db_path)
        self.cursor = self.connection.cursor()
        self._create_table()

    def _create_table(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message TEXT NOT NULL
            )
        ''')
        self.connection.commit()

    def store_message(self, message: str):
        self.cursor.execute('INSERT INTO conversation_history (message) VALUES (?)', (message,))
        self.connection.commit()

    def retrieve_messages(self):
        self.cursor.execute('SELECT message FROM conversation_history')
        return [row[0] for row in self.cursor.fetchall()]

    def close(self):
        self.connection.close()
