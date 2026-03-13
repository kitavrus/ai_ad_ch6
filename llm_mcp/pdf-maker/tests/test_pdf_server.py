import json
from unittest.mock import MagicMock

import httpx
import pytest

import pdf_server
from pdf_server import create_pdf

VALID_DOC = json.dumps({
    "title": "Test Document",
    "author": "Test Author",
    "sections": [
        {"heading": "Section 1", "content": "Some content here."},
        {"heading": "List", "items": ["Item A", "Item B"]},
    ],
})

SIMPLE_DOC = json.dumps({"title": "Simple"})

TABLE_DOC = json.dumps({
    "title": "Table Report",
    "sections": [
        {
            "heading": "Data",
            "table": {
                "headers": ["Name", "Score"],
                "rows": [["Alice", "95"], ["Bob", "87"]],
            },
        }
    ],
})


class TestCreatePdf:
    def test_happy_path_saved_to_disk(self, valid_auth):
        result = create_pdf(VALID_DOC)
        assert "сохранён" in result

    def test_happy_path_contains_filename(self, valid_auth):
        result = create_pdf(VALID_DOC)
        assert ".pdf" in result

    def test_happy_path_contains_size(self, valid_auth):
        result = create_pdf(VALID_DOC)
        assert "байт" in result

    def test_happy_path_no_base64_in_result(self, valid_auth):
        result = create_pdf(VALID_DOC)
        assert "pdf_base64" not in result

    def test_simple_document(self, valid_auth):
        result = create_pdf(SIMPLE_DOC)
        assert "сохранён" in result

    def test_table_document(self, valid_auth):
        result = create_pdf(TABLE_DOC)
        assert "сохранён" in result

    def test_result_prefix(self, valid_auth):
        result = create_pdf(SIMPLE_DOC)
        assert result.startswith("PDF создан и сохранён:")

    def test_filename_in_result(self, valid_auth):
        result = create_pdf(SIMPLE_DOC)
        assert "simple.pdf" in result


class TestInvalidInput:
    def test_invalid_json_returns_error(self, valid_auth):
        result = create_pdf("not valid json {{{")
        assert "Ошибка разбора JSON" in result

    def test_empty_title_returns_validation_error(self, valid_auth):
        result = create_pdf(json.dumps({"title": ""}))
        assert "Ошибка валидации" in result or "422" in result

    def test_missing_title_returns_validation_error(self, valid_auth):
        result = create_pdf(json.dumps({"author": "No Title"}))
        assert "Ошибка валидации" in result or "422" in result


class TestAuthError:
    def test_invalid_auth_returns_auth_error(self, invalid_auth):
        result = create_pdf(SIMPLE_DOC)
        assert "Ошибка авторизации" in result
        assert "X-API-Key" in result


class TestNetworkErrors:
    def test_connect_error_returns_message(self, monkeypatch):
        def raise_connect_error(*args, **kwargs):
            raise httpx.ConnectError("Connection refused")

        monkeypatch.setattr(pdf_server.httpx, "post", raise_connect_error)
        result = create_pdf(SIMPLE_DOC)
        assert "Не удалось подключиться" in result
        assert "localhost:8883" in result

    def test_timeout_error_returns_message(self, monkeypatch):
        def raise_timeout(*args, **kwargs):
            raise httpx.TimeoutException("Timeout", request=MagicMock())

        monkeypatch.setattr(pdf_server.httpx, "post", raise_timeout)
        result = create_pdf(SIMPLE_DOC)
        assert "Превышено время ожидания" in result
