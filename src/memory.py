"""Persistent conversation memory utilities for the Buggy Coder agent."""
from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Dict, List

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict

__all__ = [
	"PersistentChatMessageHistory",
	"get_history",
]

_DEFAULT_STORAGE_DIR = Path(
	os.environ.get("BUGGY_CODER_MEMORY_DIR", ".memory/buggy_coder"),
).resolve()


def _sanitize_session_id(session_id: str) -> str:
	"""Create a filesystem-safe representation of a session identifier."""
	if not session_id:
		return "default"
	return re.sub(r"[^a-zA-Z0-9_.-]", "_", session_id)


class PersistentChatMessageHistory(BaseChatMessageHistory):
	"""Simple JSON-backed implementation of chat message history.

	This implementation stores messages for each session in a dedicated JSON file
	on disk so that conversations persist across agent runs and threads.
	"""

	def __init__(self, session_id: str, storage_dir: Path | None = None) -> None:
		self._session_id = session_id or "default"
		self._storage_dir = (storage_dir or _DEFAULT_STORAGE_DIR).expanduser().resolve()
		self._storage_dir.mkdir(parents=True, exist_ok=True)
		sanitized = _sanitize_session_id(self._session_id)
		self._file_path = self._storage_dir / f"{sanitized}.json"
		self._lock = threading.RLock()
		self._messages: List[BaseMessage] = []
		self._load()

	@property
	def messages(self) -> List[BaseMessage]:  # type: ignore[override]
		with self._lock:
			return list(self._messages)

	def add_message(self, message: BaseMessage) -> None:  # type: ignore[override]
		with self._lock:
			self._messages.append(message)
			self._save_locked()

	def add_messages(self, messages: List[BaseMessage]) -> None:  # type: ignore[override]
		if not messages:
			return
		with self._lock:
			self._messages.extend(messages)
			self._save_locked()

	def clear(self) -> None:  # type: ignore[override]
		with self._lock:
			self._messages = []
			self._save_locked()

	def _load(self) -> None:
		if not self._file_path.exists():
			self._messages = []
			return
		try:
			with self._file_path.open("r", encoding="utf-8") as file:
				data = json.load(file)
				self._messages = messages_from_dict(data)
		except (json.JSONDecodeError, OSError, TypeError, ValueError):
			# If the file is corrupt or unreadable, reset to an empty history.
			self._messages = []

	def _save_locked(self) -> None:
		payload = messages_to_dict(self._messages)
		tmp_path = self._file_path.with_suffix(".tmp")
		with tmp_path.open("w", encoding="utf-8") as file:
			json.dump(payload, file, ensure_ascii=False, indent=2)
		tmp_path.replace(self._file_path)


_history_cache: Dict[str, PersistentChatMessageHistory] = {}
_cache_lock = threading.Lock()


def get_history(session_id: str) -> PersistentChatMessageHistory:
	"""Return a cached history instance for a session identifier."""
	key = session_id or "default"
	with _cache_lock:
		history = _history_cache.get(key)
		if history is None:
			history = PersistentChatMessageHistory(key)
			_history_cache[key] = history
	return history
