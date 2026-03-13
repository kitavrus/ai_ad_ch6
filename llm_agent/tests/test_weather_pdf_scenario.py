"""Integration tests: Weather → PDF → Save scenario.

Requires running FastAPI servers on localhost:8882 (weather), 8883 (pdf-maker), 8884 (save-to-file).

Run:
    cd llm_agent
    python -m pytest tests/test_weather_pdf_scenario.py -v -m integration
"""

import json
import time
from unittest.mock import MagicMock, patch

import httpx
import pytest

from llm_agent.chatbot.main import _handle_tool_calls
from llm_agent.chatbot.memory import Memory
from llm_agent.chatbot.models import SessionState
from llm_agent.chatbot.mcp_client import (
    MCPClientManager,
    MCPPdfMakerClient,
    MCPSaveToFileClient,
    MCPWeatherClient,
)

pytestmark = pytest.mark.integration

# A minimal fixed document for create_pdf (weather content populated by real API calls in direct test)
_FIXED_DOC_JSON = json.dumps({
    "title": "Погода в городах России",
    "author": "Weather Bot",
    "sections": [
        {
            "heading": "Обзор",
            "content": "Тестовый отчёт о погоде в Москве, Санкт-Петербурге и Казани.",
        }
    ],
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _servers_available() -> bool:
    """Return True only if all three FastAPI servers respond."""
    for port in [8882, 8883, 8884]:
        try:
            httpx.get(f"http://localhost:{port}/", timeout=2.0)
        except Exception:
            return False
    return True


def _make_state() -> SessionState:
    state = SessionState(
        model="test-model",
        base_url="http://localhost",
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        dialogue_start_time=time.time(),
    )
    state.memory = Memory()
    return state


def _make_tool_call_mock(tc_id: str, name: str, args: dict) -> MagicMock:
    tc = MagicMock()
    tc.id = tc_id
    tc.function.name = name
    tc.function.arguments = json.dumps(args)
    return tc


def _make_response(tool_calls=None, content: str = "") -> MagicMock:
    resp = MagicMock()
    resp.choices[0].message.tool_calls = tool_calls
    resp.choices[0].message.content = content
    return resp


# ---------------------------------------------------------------------------
# Shared fixture: MCPClientManager connected to all three servers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mcp_manager():
    if not _servers_available():
        pytest.skip("MCP/API servers not available (run on localhost:8882-8884)")
    manager = MCPClientManager([
        MCPWeatherClient(),
        MCPPdfMakerClient(),
        MCPSaveToFileClient(),
    ])
    results = manager.connect_all()
    if not all(results.values()):
        pytest.skip("One or more MCP clients failed to connect")
    return manager


# ---------------------------------------------------------------------------
# Test A: direct tool chain without LLM
# ---------------------------------------------------------------------------

class TestToolChainDirect:
    def test_tool_chain_direct(self, mcp_manager):
        """Call get_weather × 3 → create_pdf → save_file using real API servers."""

        # Step 1: weather for all three cities
        moscow = mcp_manager.call_tool("get_weather", {"city": "Москва"})
        spb = mcp_manager.call_tool("get_weather", {"city": "Санкт-Петербург"})
        kazan = mcp_manager.call_tool("get_weather", {"city": "Казань"})

        assert "Москва" in moscow, f"Expected 'Москва' in: {moscow}"
        assert "Санкт-Петербург" in spb, f"Expected 'Санкт-Петербург' in: {spb}"
        assert "Казань" in kazan, f"Expected 'Казань' in: {kazan}"

        # Step 2: create PDF with real weather data
        doc = {
            "title": "Погода в городах России",
            "author": "Test Bot",
            "sections": [
                {"heading": "Москва", "content": moscow},
                {"heading": "Санкт-Петербург", "content": spb},
                {"heading": "Казань", "content": kazan},
            ],
        }
        pdf_result = mcp_manager.call_tool("create_pdf", {"document_json": json.dumps(doc)})

        assert "PDF создан:" in pdf_result, f"Expected 'PDF создан:' in: {pdf_result}"
        assert "pdf_base64:" in pdf_result, f"Expected 'pdf_base64:' in: {pdf_result}"

        # Extract filename and base64 from result string
        first_line = pdf_result.split("\n")[0]  # "PDF создан: <name> (<size> байт)"
        filename = first_line.split("PDF создан: ")[1].split(" (")[0]
        pdf_b64 = pdf_result.split("pdf_base64: ")[1].strip()

        assert filename.endswith(".pdf"), f"Expected .pdf filename, got: {filename}"
        assert len(pdf_b64) > 0, "pdf_base64 should not be empty"

        # Step 3: save the PDF file
        save_result = mcp_manager.call_tool(
            "save_file", {"filename": filename, "content_base64": pdf_b64}
        )

        assert "Файл сохранён:" in save_result, f"Expected 'Файл сохранён:' in: {save_result}"


# ---------------------------------------------------------------------------
# Test B: full orchestration via _handle_tool_calls with mock LLM
# ---------------------------------------------------------------------------

class TestHandleToolCallsWeatherPdfSave:
    def test_handle_tool_calls_weather_pdf_save(self, mcp_manager, monkeypatch, tmp_path):
        """LLM orchestrates: 3×get_weather → create_pdf → save_file.

        The LLM is mocked; MCP tool calls hit real FastAPI servers.
        The save_file call is built dynamically using the real pdf_base64 that
        create_pdf returns.
        """
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        api_messages = [{"role": "user", "content": "Составь отчёт о погоде в трёх городах"}]

        # Initial response from LLM: three parallel weather tool calls
        resp1 = _make_response(tool_calls=[
            _make_tool_call_mock("c1", "get_weather", {"city": "Москва"}),
            _make_tool_call_mock("c2", "get_weather", {"city": "Санкт-Петербург"}),
            _make_tool_call_mock("c3", "get_weather", {"city": "Казань"}),
        ])

        # Second LLM response: create_pdf with a fixed document structure
        resp2 = _make_response(tool_calls=[
            _make_tool_call_mock("c4", "create_pdf", {"document_json": _FIXED_DOC_JSON}),
        ])

        # Final LLM response: plain text, no tool calls
        resp4 = _make_response(content="PDF с погодой для трёх городов сохранён.")

        call_count = 0

        def llm_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                # After 3 weather calls → ask LLM to create PDF
                return resp2

            if call_count == 2:
                # After create_pdf → extract real pdf_base64 and build save_file call
                messages = kwargs.get("messages", [])
                pdf_b64 = ""
                filename = "weather_report.pdf"
                for msg in reversed(messages):
                    content = msg.get("content", "")
                    if msg.get("role") == "tool" and "pdf_base64:" in content:
                        first_line = content.split("\n")[0]
                        filename = first_line.split("PDF создан: ")[1].split(" (")[0]
                        pdf_b64 = content.split("pdf_base64: ")[1].strip()
                        break

                resp3 = _make_response(tool_calls=[
                    _make_tool_call_mock(
                        "c5", "save_file",
                        {"filename": filename, "content_base64": pdf_b64},
                    ),
                ])
                return resp3

            # call_count == 3: after save_file → final answer
            return resp4

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = llm_side_effect

        with patch.object(mcp_manager, "call_tool", wraps=mcp_manager.call_tool) as spy:
            final_response, text = _handle_tool_calls(
                resp1, api_messages, state, mock_client, mcp_manager, {},
                notification_server=None,
            )

        # --- Assertions ---

        # Final text must match what the mock LLM returned
        assert text == "PDF с погодой для трёх городов сохранён."

        # Collect all tool names called
        tool_names = [c.args[0] for c in spy.call_args_list]

        # All three cities were queried
        assert tool_names.count("get_weather") == 3, f"Expected 3 get_weather calls, got: {tool_names}"

        # PDF was created
        assert "create_pdf" in tool_names, f"create_pdf not called: {tool_names}"

        # File was saved
        assert "save_file" in tool_names, f"save_file not called: {tool_names}"

        # save_file received a non-empty base64 payload
        save_calls = [c for c in spy.call_args_list if c.args[0] == "save_file"]
        assert len(save_calls) == 1
        save_args = save_calls[0].args[1]
        assert "content_base64" in save_args
        assert len(save_args["content_base64"]) > 0, "save_file called with empty base64"

        # The real save_file API confirmed the file was saved (check tool result in api_messages)
        save_results = [
            msg["content"]
            for msg in api_messages
            if msg.get("role") == "tool" and "Файл сохранён:" in msg.get("content", "")
        ]
        assert save_results, "No 'Файл сохранён:' confirmation found in api_messages"
