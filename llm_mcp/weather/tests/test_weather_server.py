from unittest.mock import MagicMock, patch

import httpx
import pytest

import weather_server
from weather_server import get_weather, list_cities


class TestGetWeatherSuccess:
    def test_russian_city_format(self, valid_auth):
        result = get_weather("москва")
        assert "Погода в Москва:" in result
        assert "Температура:" in result
        assert "Состояние:" in result
        assert "Влажность:" in result
        assert "Ветер:" in result

    def test_russian_city_units(self, valid_auth):
        result = get_weather("москва")
        assert "°C" in result
        assert "%" in result
        assert "м/с" in result

    def test_english_alias_moscow(self, valid_auth):
        result = get_weather("moscow")
        assert "Москва" in result

    def test_english_alias_novosibirsk(self, valid_auth):
        result = get_weather("novosibirsk")
        assert "Новосибирск" in result

    def test_uppercase_city(self, valid_auth):
        result = get_weather("МОСКВА")
        assert "Москва" in result

    def test_russian_alias_msk(self, valid_auth):
        result = get_weather("мск")
        assert "Москва" in result


class TestGetWeatherErrors:
    def test_unknown_city(self, valid_auth):
        result = get_weather("london")
        assert "london" in result.lower() or "не найден" in result

    def test_invalid_auth(self, invalid_auth):
        result = get_weather("москва")
        assert "Ошибка авторизации" in result
        assert "X-API-Key" in result


class TestGetWeatherNetworkErrors:
    def test_connect_error(self, monkeypatch):
        def raise_connect_error(*args, **kwargs):
            raise httpx.ConnectError("Connection refused")

        monkeypatch.setattr(weather_server.httpx, "get", raise_connect_error)
        result = get_weather("москва")
        assert "Не удалось подключиться" in result
        assert "localhost:8000" in result

    def test_timeout_error(self, monkeypatch):
        def raise_timeout(*args, **kwargs):
            raise httpx.TimeoutException("Timeout", request=MagicMock())

        monkeypatch.setattr(weather_server.httpx, "get", raise_timeout)
        result = get_weather("москва")
        assert "Превышено время ожидания" in result


class TestListCitiesSuccess:
    def test_starts_with_header(self, valid_auth):
        result = list_cities()
        assert result.startswith("Доступные города:")

    def test_contains_moscow(self, valid_auth):
        result = list_cities()
        assert "Москва" in result

    def test_contains_saint_petersburg(self, valid_auth):
        result = list_cities()
        assert "Санкт-Петербург" in result

    def test_exactly_35_cities(self, valid_auth):
        result = list_cities()
        city_lines = [line for line in result.splitlines() if line.startswith("  - ")]
        assert len(city_lines) == 35

    def test_returns_str(self, valid_auth):
        result = list_cities()
        assert isinstance(result, str)


class TestListCitiesErrors:
    def test_invalid_auth(self, invalid_auth):
        result = list_cities()
        assert "Ошибка авторизации" in result
        assert "X-API-Key" in result


class TestListCitiesNetworkErrors:
    def test_connect_error(self, monkeypatch):
        def raise_connect_error(*args, **kwargs):
            raise httpx.ConnectError("Connection refused")

        monkeypatch.setattr(weather_server.httpx, "get", raise_connect_error)
        result = list_cities()
        assert "Не удалось подключиться" in result

    def test_timeout_error(self, monkeypatch):
        def raise_timeout(*args, **kwargs):
            raise httpx.TimeoutException("Timeout", request=MagicMock())

        monkeypatch.setattr(weather_server.httpx, "get", raise_timeout)
        result = list_cities()
        assert "Превышено время ожидания" in result
