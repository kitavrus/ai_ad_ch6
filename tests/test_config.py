"""Тесты для модуля chatbot.config."""

import pytest
from pydantic import ValidationError

from chatbot.config import (
    BASE_URL,
    CONTEXT_RECENT_MESSAGES,
    CONTEXT_SUMMARY_INTERVAL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    DIALOGUES_DIR,
    METRICS_DIR,
    RUB_PER_USD,
    USD_PER_1K_TOKENS,
    SessionConfig,
)


class TestConstants:
    def test_base_url(self):
        assert BASE_URL == "https://routerai.ru/api/v1"

    def test_default_model(self):
        assert DEFAULT_MODEL == "inception/mercury-coder"

    def test_default_max_tokens_is_none(self):
        assert DEFAULT_MAX_TOKENS is None

    def test_default_temperature(self):
        assert DEFAULT_TEMPERATURE == 0.7

    def test_default_top_p(self):
        assert DEFAULT_TOP_P == 0.9

    def test_default_top_k(self):
        assert DEFAULT_TOP_K == 50

    def test_usd_per_1k_tokens(self):
        assert USD_PER_1K_TOKENS == 0.0015

    def test_rub_per_usd(self):
        assert RUB_PER_USD == 100.0

    def test_context_recent_messages(self):
        assert CONTEXT_RECENT_MESSAGES == 10

    def test_context_summary_interval(self):
        assert CONTEXT_SUMMARY_INTERVAL == 10

    def test_dialogues_dir(self):
        assert DIALOGUES_DIR == "dialogues"

    def test_metrics_dir(self):
        assert METRICS_DIR == "dialogues/metrics"


class TestSessionConfig:
    def test_defaults(self):
        cfg = SessionConfig()
        assert cfg.model == DEFAULT_MODEL
        assert cfg.base_url == BASE_URL
        assert cfg.max_tokens is None
        assert cfg.temperature == DEFAULT_TEMPERATURE
        assert cfg.top_p == DEFAULT_TOP_P
        assert cfg.top_k == DEFAULT_TOP_K
        assert cfg.system_prompt is None
        assert cfg.initial_prompt is None

    def test_custom_values(self):
        cfg = SessionConfig(
            model="gpt-4",
            base_url="https://api.example.com",
            max_tokens=512,
            temperature=0.5,
            top_p=0.8,
            top_k=20,
            system_prompt="You are helpful.",
            initial_prompt="Hello!",
        )
        assert cfg.model == "gpt-4"
        assert cfg.base_url == "https://api.example.com"
        assert cfg.max_tokens == 512
        assert cfg.temperature == 0.5
        assert cfg.top_p == 0.8
        assert cfg.top_k == 20
        assert cfg.system_prompt == "You are helpful."
        assert cfg.initial_prompt == "Hello!"

    def test_temperature_ge_zero(self):
        with pytest.raises(ValidationError):
            SessionConfig(temperature=-0.1)

    def test_temperature_le_two(self):
        with pytest.raises(ValidationError):
            SessionConfig(temperature=2.1)

    def test_top_p_ge_zero(self):
        with pytest.raises(ValidationError):
            SessionConfig(top_p=-0.1)

    def test_top_p_le_one(self):
        with pytest.raises(ValidationError):
            SessionConfig(top_p=1.1)

    def test_top_k_ge_zero(self):
        with pytest.raises(ValidationError):
            SessionConfig(top_k=-1)

    def test_temperature_boundary_zero(self):
        cfg = SessionConfig(temperature=0.0)
        assert cfg.temperature == 0.0

    def test_temperature_boundary_two(self):
        cfg = SessionConfig(temperature=2.0)
        assert cfg.temperature == 2.0

    def test_top_p_boundary_zero(self):
        cfg = SessionConfig(top_p=0.0)
        assert cfg.top_p == 0.0

    def test_top_p_boundary_one(self):
        cfg = SessionConfig(top_p=1.0)
        assert cfg.top_p == 1.0

    def test_top_k_zero_allowed(self):
        cfg = SessionConfig(top_k=0)
        assert cfg.top_k == 0
