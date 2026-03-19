"""Tests for new /rag interactive commands."""
import pytest
from unittest.mock import MagicMock, patch

from chatbot.models import RagMode, SessionState
from chatbot.main import _handle_rag_command, _get_retriever


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(**kwargs) -> SessionState:
    base = dict(
        model="test-model",
        base_url="http://localhost",
        temperature=0.7,
        top_p=1.0,
        top_k=3,
    )
    base.update(kwargs)
    return SessionState(**base)


def _make_result(score=0.8, distance=0.2, title="T", section="S", text="some text", query="q"):
    r = MagicMock()
    r.score = score
    r.distance = distance
    r.chunk.title = title
    r.chunk.section = section
    r.chunk.text = text
    r.query = query
    return r


# ---------------------------------------------------------------------------
# /rag filter
# ---------------------------------------------------------------------------

def test_filter_sets_threshold(capsys):
    state = _make_state()
    _handle_rag_command("filter", "0.5", state)
    assert state.rag_mode.threshold == 0.5
    assert "0.5" in capsys.readouterr().out


def test_filter_clamps_below_zero(capsys):
    state = _make_state()
    _handle_rag_command("filter", "-0.1", state)
    assert state.rag_mode.threshold == 0.0


def test_filter_clamps_above_one(capsys):
    state = _make_state()
    _handle_rag_command("filter", "1.5", state)
    assert state.rag_mode.threshold == 1.0


def test_filter_bad_value(capsys):
    state = _make_state()
    _handle_rag_command("filter", "abc", state)
    assert state.rag_mode.threshold == 0.0
    assert "требует" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# /rag topk_before / topk_after
# ---------------------------------------------------------------------------

def test_topk_before(capsys):
    state = _make_state()
    _handle_rag_command("topk_before", "10", state)
    assert state.rag_mode.top_k_before == 10
    out = capsys.readouterr().out
    assert "topk_before" in out


def test_topk_after(capsys):
    state = _make_state()
    _handle_rag_command("topk_after", "5", state)
    assert state.rag_mode.top_k_after == 5


def test_topk_before_bad_value(capsys):
    state = _make_state()
    _handle_rag_command("topk_before", "xyz", state)
    assert "требует" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# /rag rewrite
# ---------------------------------------------------------------------------

def test_rewrite_on(capsys):
    state = _make_state()
    _handle_rag_command("rewrite", "on", state)
    assert state.rag_mode.rewrite_query is True
    assert "ON" in capsys.readouterr().out


def test_rewrite_off(capsys):
    state = _make_state()
    state.rag_mode.rewrite_query = True
    _handle_rag_command("rewrite", "off", state)
    assert state.rag_mode.rewrite_query is False
    assert "OFF" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# /rag mode B|C|D
# ---------------------------------------------------------------------------

def test_mode_B(capsys):
    state = _make_state()
    state.rag_mode.threshold = 0.7
    state.rag_mode.rewrite_query = True
    _handle_rag_command("mode", "B", state)
    assert state.rag_mode.threshold == 0.0
    assert state.rag_mode.top_k_before == 0
    assert state.rag_mode.top_k_after == 0
    assert state.rag_mode.rewrite_query is False
    assert "B" in capsys.readouterr().out


def test_mode_C(capsys):
    state = _make_state()
    _handle_rag_command("mode", "C", state)
    assert state.rag_mode.threshold == 0.5
    assert state.rag_mode.top_k_before == 10
    assert state.rag_mode.top_k_after == 3
    assert state.rag_mode.rewrite_query is False
    assert "C" in capsys.readouterr().out


def test_mode_D(capsys):
    state = _make_state()
    _handle_rag_command("mode", "D", state)
    assert state.rag_mode.threshold == 0.5
    assert state.rag_mode.top_k_before == 10
    assert state.rag_mode.top_k_after == 3
    assert state.rag_mode.rewrite_query is True
    assert "D" in capsys.readouterr().out


def test_mode_lowercase(capsys):
    state = _make_state()
    _handle_rag_command("mode", "d", state)
    assert state.rag_mode.rewrite_query is True


def test_mode_unknown(capsys):
    state = _make_state()
    _handle_rag_command("mode", "X", state)
    assert "требует" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# /rag status — extended output
# ---------------------------------------------------------------------------

def test_status_shows_new_fields(capsys):
    state = _make_state()
    state.rag_mode.threshold = 0.5
    state.rag_mode.top_k_before = 10
    state.rag_mode.top_k_after = 3
    state.rag_mode.rewrite_query = True
    _handle_rag_command("status", "", state)
    out = capsys.readouterr().out
    assert "0.5" in out
    assert "10" in out
    assert "ON" in out


# ---------------------------------------------------------------------------
# /rag search
# ---------------------------------------------------------------------------

def test_search_calls_retriever(capsys):
    state = _make_state()
    mock_ret = MagicMock()
    mock_ret.search_with_scores.return_value = [_make_result()]
    with patch("chatbot.main._get_retriever", return_value=mock_ret):
        _handle_rag_command("search", "test query", state)
    mock_ret.search_with_scores.assert_called_once_with(
        "test query", strategy="structure", top_k=3
    )
    out = capsys.readouterr().out
    assert "T / S" in out


def test_search_no_results(capsys):
    state = _make_state()
    mock_ret = MagicMock()
    mock_ret.search_with_scores.return_value = []
    with patch("chatbot.main._get_retriever", return_value=mock_ret):
        _handle_rag_command("search", "nothing", state)
    assert "не найдены" in capsys.readouterr().out


def test_search_empty_arg(capsys):
    state = _make_state()
    _handle_rag_command("search", "", state)
    assert "требует" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# /rag compare
# ---------------------------------------------------------------------------

def test_compare_three_modes(capsys):
    state = _make_state()

    def make_retriever(**kwargs):
        mock_ret = MagicMock()
        mock_ret.search_with_scores.return_value = [_make_result()]
        return mock_ret

    with patch("llm_agent.rag.retriever.RAGRetriever", side_effect=make_retriever):
        _handle_rag_command("compare", "test query", state)

    out = capsys.readouterr().out
    assert "Режим B" in out
    assert "Режим C" in out
    assert "Режим D" in out


def test_compare_empty_arg(capsys):
    state = _make_state()
    _handle_rag_command("compare", "", state)
    assert "требует" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# _get_retriever — config caching
# ---------------------------------------------------------------------------

def test_get_retriever_recreates_on_config_change():
    import chatbot.main as m
    m._retriever = None
    m._retriever_config = ()

    rag1 = RagMode(threshold=0.0, rewrite_query=False)
    rag2 = RagMode(threshold=0.5, rewrite_query=False)

    with patch("llm_agent.rag.retriever.RAGRetriever") as MockR:
        MockR.return_value = MagicMock()
        r1 = m._get_retriever(rag1)
        r2 = m._get_retriever(rag1)   # same config → same instance
        assert r1 is r2
        assert MockR.call_count == 1

        r3 = m._get_retriever(rag2)   # changed config → new instance
        assert MockR.call_count == 2

    m._retriever = None
    m._retriever_config = ()


def test_get_retriever_no_rag_mode():
    import chatbot.main as m
    m._retriever = None
    m._retriever_config = ()
    with patch("llm_agent.rag.retriever.RAGRetriever") as MockR:
        MockR.return_value = MagicMock()
        m._get_retriever(None)
        assert MockR.call_count == 1
    m._retriever = None
    m._retriever_config = ()
