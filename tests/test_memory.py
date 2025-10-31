import json
import os
import re
import sys
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT / "src"))

import memory
from memory import PersistentChatMessageHistory, get_history


@pytest.fixture(autouse=True)
def reset_history_cache():
	memory._history_cache.clear()
	yield
	memory._history_cache.clear()


@pytest.fixture
def temp_storage_dir(tmp_path, monkeypatch):
	monkeypatch.setenv("BUGGY_CODER_MEMORY_DIR", str(tmp_path))
	monkeypatch.setattr(memory, "_DEFAULT_STORAGE_DIR", tmp_path)
	return tmp_path


def test_persistent_history_persists_messages(temp_storage_dir):
	history = PersistentChatMessageHistory("thread-123", storage_dir=temp_storage_dir)
	history.add_messages([HumanMessage(content="hello"), AIMessage(content="hi there")])

	# Simulate a new agent/thread loading the same history.
	reloaded = PersistentChatMessageHistory("thread-123", storage_dir=temp_storage_dir)
	contents = [message.content for message in reloaded.messages]

	assert contents == ["hello", "hi there"]


def test_clear_resets_history_and_file(temp_storage_dir):
	history = PersistentChatMessageHistory("thread-456", storage_dir=temp_storage_dir)
	history.add_message(HumanMessage(content="remember me"))

	history.clear()

	assert history.messages == []

	file_contents = json.loads((temp_storage_dir / "thread-456.json").read_text("utf-8"))
	assert file_contents == []


def test_session_id_sanitization(temp_storage_dir):
	session_id = "user/with spaces*&"
	expected_file = re.sub(r"[^a-zA-Z0-9_.-]", "_", session_id) + ".json"

	history = PersistentChatMessageHistory(session_id, storage_dir=temp_storage_dir)

	assert history._file_path.name == expected_file


def test_get_history_caches_instances(temp_storage_dir):
	history_one = get_history("shared-thread")
	history_two = get_history("shared-thread")
	another_history = get_history("different-thread")

	assert history_one is history_two
	assert history_one is not another_history


def test_corrupt_history_file_is_handled(temp_storage_dir):
	session_id = "problem-session"
	sanitized = re.sub(r"[^a-zA-Z0-9_.-]", "_", session_id)
	history_file = temp_storage_dir / f"{sanitized}.json"
	history_file.write_text("{this is not valid json}", "utf-8")

	history = PersistentChatMessageHistory(session_id, storage_dir=temp_storage_dir)

	assert history.messages == []
	assert history_file.exists()
