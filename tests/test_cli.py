"""Тесты для модуля chatbot.cli."""

import argparse
import sys

import pytest

from chatbot.cli import (
    config_from_args,
    get_resume_flag,
    parse_args,
    parse_inline_command,
)
from chatbot.config import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
)


def _make_args(**kwargs) -> argparse.Namespace:
    """Создаёт Namespace с дефолтными значениями, перекрытыми kwargs."""
    defaults = {
        "model": None,
        "base_url": None,
        "max_tokens": None,
        "temperature": None,
        "top_p": None,
        "top_k": None,
        "system_prompt": None,
        "initial_prompt": None,
        "resume": False,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


# ---------------------------------------------------------------------------
# config_from_args
# ---------------------------------------------------------------------------


class TestConfigFromArgs:
    def test_defaults_when_all_none(self):
        args = _make_args()
        cfg = config_from_args(args)
        assert cfg.model == DEFAULT_MODEL
        assert cfg.max_tokens == DEFAULT_MAX_TOKENS
        assert cfg.temperature == DEFAULT_TEMPERATURE
        assert cfg.top_p == DEFAULT_TOP_P
        assert cfg.top_k == DEFAULT_TOP_K
        assert cfg.system_prompt is None
        assert cfg.initial_prompt is None

    def test_custom_model(self):
        args = _make_args(model="gpt-4")
        cfg = config_from_args(args)
        assert cfg.model == "gpt-4"

    def test_custom_temperature(self):
        args = _make_args(temperature=0.3)
        cfg = config_from_args(args)
        assert cfg.temperature == 0.3

    def test_custom_top_p(self):
        args = _make_args(top_p=0.5)
        cfg = config_from_args(args)
        assert cfg.top_p == 0.5

    def test_custom_top_k(self):
        args = _make_args(top_k=10)
        cfg = config_from_args(args)
        assert cfg.top_k == 10

    def test_custom_max_tokens(self):
        args = _make_args(max_tokens=512)
        cfg = config_from_args(args)
        assert cfg.max_tokens == 512

    def test_system_prompt(self):
        args = _make_args(system_prompt="You are a robot.")
        cfg = config_from_args(args)
        assert cfg.system_prompt == "You are a robot."

    def test_initial_prompt(self):
        args = _make_args(initial_prompt="Start now.")
        cfg = config_from_args(args)
        assert cfg.initial_prompt == "Start now."


# ---------------------------------------------------------------------------
# get_resume_flag
# ---------------------------------------------------------------------------


class TestGetResumeFlag:
    def test_false_by_default(self):
        args = _make_args()
        assert get_resume_flag(args) is False

    def test_true_when_set(self):
        args = _make_args(resume=True)
        assert get_resume_flag(args) is True

    def test_missing_attribute_returns_false(self):
        args = argparse.Namespace()
        assert get_resume_flag(args) is False


# ---------------------------------------------------------------------------
# parse_inline_command
# ---------------------------------------------------------------------------


class TestParseInlineCommand:
    # --- Non-command inputs ---

    def test_empty_string(self):
        assert parse_inline_command("") == {}

    def test_no_slash_returns_empty(self):
        assert parse_inline_command("hello world") == {}

    def test_slash_only(self):
        assert parse_inline_command("/") == {}

    def test_slash_with_spaces_only(self):
        assert parse_inline_command("/   ") == {}

    # --- model ---

    def test_model_with_equals(self):
        result = parse_inline_command("/model=gpt-4")
        assert result == {"model": "gpt-4"}

    def test_model_with_space(self):
        result = parse_inline_command("/model gpt-4")
        assert result == {"model": "gpt-4"}

    def test_model_empty_value(self):
        result = parse_inline_command("/model=")
        assert result["model"] is None

    # --- base_url ---

    def test_base_url_with_equals(self):
        result = parse_inline_command("/base-url=https://api.example.com")
        assert result == {"base_url": "https://api.example.com"}

    def test_base_url_with_space(self):
        result = parse_inline_command("/base-url https://api.example.com")
        assert result == {"base_url": "https://api.example.com"}

    def test_baseurl_alias(self):
        result = parse_inline_command("/baseurl https://example.com")
        assert result == {"base_url": "https://example.com"}

    def test_base_url_empty(self):
        result = parse_inline_command("/base-url=")
        assert result["base_url"] is None

    # --- max_tokens ---

    def test_max_tokens_valid(self):
        result = parse_inline_command("/max-tokens 1024")
        assert result == {"max_tokens": 1024}

    def test_max_tokens_invalid(self):
        result = parse_inline_command("/max-tokens abc")
        assert result == {"max_tokens": None}

    def test_max_tokens_with_equals(self):
        result = parse_inline_command("/max-tokens=512")
        assert result == {"max_tokens": 512}

    # --- temperature ---

    def test_temperature_valid(self):
        result = parse_inline_command("/temperature 0.8")
        assert result == {"temperature": 0.8}

    def test_temperature_invalid(self):
        result = parse_inline_command("/temperature hot")
        assert result == {"temperature": None}

    def test_temperature_with_equals(self):
        result = parse_inline_command("/temperature=1.0")
        assert result == {"temperature": 1.0}

    # --- top_p ---

    def test_top_p_valid(self):
        result = parse_inline_command("/top-p 0.92")
        assert result == {"top_p": 0.92}

    def test_top_p_invalid(self):
        result = parse_inline_command("/top-p xyz")
        assert result == {"top_p": None}

    # --- top_k ---

    def test_top_k_valid(self):
        result = parse_inline_command("/top-k 50")
        assert result == {"top_k": 50}

    def test_top_k_invalid(self):
        result = parse_inline_command("/top-k notanumber")
        assert result == {"top_k": None}

    # --- system_prompt ---

    def test_system_prompt_with_space(self):
        result = parse_inline_command("/system-prompt You are a bot.")
        assert result == {"system_prompt": "You are a bot."}

    def test_system_prompt_underscore_alias(self):
        result = parse_inline_command("/system_prompt Be concise.")
        assert result == {"system_prompt": "Be concise."}

    # --- initial_prompt ---

    def test_initial_prompt(self):
        result = parse_inline_command("/initial-prompt Hello there!")
        assert result == {"initial_prompt": "Hello there!"}

    def test_initial_prompt_underscore(self):
        result = parse_inline_command("/initial_prompt Start.")
        assert result == {"initial_prompt": "Start."}

    # --- resume ---

    def test_resume_true(self):
        result = parse_inline_command("/resume true")
        assert result == {"resume": True}

    def test_resume_yes(self):
        result = parse_inline_command("/resume yes")
        assert result == {"resume": True}

    def test_resume_1(self):
        result = parse_inline_command("/resume 1")
        assert result == {"resume": True}

    def test_resume_on(self):
        result = parse_inline_command("/resume on")
        assert result == {"resume": True}

    def test_resume_false(self):
        result = parse_inline_command("/resume false")
        assert result == {"resume": False}

    def test_resume_no(self):
        result = parse_inline_command("/resume no")
        assert result == {"resume": False}

    # --- showsummary ---

    def test_showsummary(self):
        result = parse_inline_command("/showsummary")
        assert result == {"showsummary": True}

    def test_showsummary_with_value(self):
        result = parse_inline_command("/showsummary anything")
        assert result == {"showsummary": True}

    # --- unknown command ---

    def test_unknown_command_returns_empty(self):
        result = parse_inline_command("/unknowncommand value")
        assert result == {}

    # --- case normalization ---

    def test_key_case_insensitive(self):
        result = parse_inline_command("/MODEL=gpt-4")
        assert result == {"model": "gpt-4"}

    def test_key_with_leading_spaces(self):
        result = parse_inline_command("  /model=gpt-4  ")
        assert result == {"model": "gpt-4"}


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------


class TestParseArgs:
    def test_defaults(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["prog"])
        args = parse_args()
        assert args.model is None
        assert args.base_url is None
        assert args.max_tokens is None
        assert args.temperature is None
        assert args.top_p is None
        assert args.top_k is None
        assert args.system_prompt is None
        assert args.initial_prompt is None
        assert args.resume is False

    def test_model_flag(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["prog", "-m", "gpt-4"])
        args = parse_args()
        assert args.model == "gpt-4"

    def test_resume_flag(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["prog", "--resume"])
        args = parse_args()
        assert args.resume is True

    def test_max_tokens_flag(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["prog", "--max-tokens", "512"])
        args = parse_args()
        assert args.max_tokens == 512

    def test_temperature_flag(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["prog", "-T", "0.5"])
        args = parse_args()
        assert args.temperature == 0.5

    def test_top_p_flag(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["prog", "-p", "0.8"])
        args = parse_args()
        assert args.top_p == 0.8

    def test_top_k_flag(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["prog", "-k", "30"])
        args = parse_args()
        assert args.top_k == 30

    def test_system_prompt_flag(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["prog", "--system-prompt", "Be helpful."])
        args = parse_args()
        assert args.system_prompt == "Be helpful."

    def test_initial_prompt_flag(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["prog", "--initial-prompt", "Hello!"])
        args = parse_args()
        assert args.initial_prompt == "Hello!"
