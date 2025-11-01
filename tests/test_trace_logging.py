import json
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.graph import TraceLoggingCallback


class TraceLoggingCallbackTests(unittest.TestCase):
    def test_trace_logging_callback_records_events(self) -> None:
        with TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "trace.log"
            callback = TraceLoggingCallback(log_path=log_path)

            callback.on_chain_start({"name": "test_chain"}, {"input": "value"})
            callback.on_llm_start({"name": "llm"}, ["prompt 1"])
            callback.on_llm_end({"response": "ok"})
            callback.on_tool_start({"name": "tool"}, "{\"arg\": 1}")
            callback.on_tool_end({"result": "done"})
            callback.on_agent_action(types.SimpleNamespace(log="did something"))
            callback.on_agent_finish(types.SimpleNamespace(return_values={"output": "final"}))
            callback.on_chain_end({"output": "answer"})

            with log_path.open("r", encoding="utf-8") as handle:
                records = [json.loads(line) for line in handle if line.strip()]

        events = [record.get("event") for record in records]
        self.assertEqual(
            events,
            [
                "chain_start",
                "llm_start",
                "llm_end",
                "tool_start",
                "tool_end",
                "agent_action",
                "agent_finish",
                "chain_end",
            ],
        )

        chain_start_payload = records[0]["payload"]
        self.assertIn("inputs", chain_start_payload)
        self.assertEqual(chain_start_payload["inputs"], {"input": "value"})
        self.assertEqual(records[1]["payload"]["prompts"], ["prompt 1"])
        self.assertEqual(records[-1]["payload"], {"outputs": {"output": "answer"}})


if __name__ == "__main__":
    unittest.main()
