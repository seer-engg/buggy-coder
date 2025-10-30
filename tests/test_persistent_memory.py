import importlib
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def import_graph_module():
    try:
        return importlib.import_module("src.graph")
    except ModuleNotFoundError as exc:
        pytest.fail(
            "Could not import 'src.graph' because a required dependency is missing: "
            f"{exc}"
        )


def test_storage_directory_is_initialized():
    graph = import_graph_module()

    assert graph._STORAGE_DIR.is_dir(), "Expected persistent storage directory to exist"
    assert graph._MEMORY_PATH.parent == graph._STORAGE_DIR
    assert graph._MEMORY_PATH.suffix == ".sqlite"


_def_message = "Agent should expose a persistent memory checkpointer"


def test_agent_uses_sqlite_checkpointer():
    graph = import_graph_module()

    from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: PLC0401

    assert isinstance(graph.MEMORY, SqliteSaver), _def_message
