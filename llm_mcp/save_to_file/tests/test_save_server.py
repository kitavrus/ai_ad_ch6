import base64
from unittest.mock import MagicMock

import httpx
import pytest

import save_server
from save_server import save_file

SIMPLE_CONTENT = base64.b64encode(b"Hello, World!").decode()
PDF_CONTENT = base64.b64encode(b"%PDF-1.4 fake pdf content").decode()


class TestSaveFile:
    def test_success_contains_saved_prefix(self, valid_auth):
        result = save_file("test.txt", SIMPLE_CONTENT)
        assert result.startswith("Файл сохранён:")

    def test_success_contains_filename(self, valid_auth):
        result = save_file("myfile.txt", SIMPLE_CONTENT)
        assert "myfile.txt" in result

    def test_success_contains_bytes(self, valid_auth):
        result = save_file("test.txt", SIMPLE_CONTENT)
        assert "байт" in result

    def test_with_subfolder(self, valid_auth):
        result = save_file("sub.txt", SIMPLE_CONTENT, subfolder="docs")
        assert result.startswith("Файл сохранён:")
        assert "docs" in result

    def test_without_subfolder(self, valid_auth):
        result = save_file("nosub.txt", SIMPLE_CONTENT)
        assert result.startswith("Файл сохранён:")

    def test_pdf_content(self, valid_auth):
        result = save_file("document.pdf", PDF_CONTENT)
        assert result.startswith("Файл сохранён:")
        assert "байт" in result


class TestAuthError:
    def test_invalid_auth_returns_auth_error(self, invalid_auth):
        result = save_file("file.txt", SIMPLE_CONTENT)
        assert "Ошибка авторизации" in result
        assert "X-API-Key" in result


class TestValidationErrors:
    def test_empty_filename_returns_validation_error(self, valid_auth):
        result = save_file("", SIMPLE_CONTENT)
        assert "Ошибка валидации" in result

    def test_invalid_base64_returns_validation_error(self, valid_auth):
        result = save_file("file.txt", "not-valid-base64!!!")
        assert "Ошибка валидации" in result

    def test_path_traversal_returns_validation_error(self, valid_auth):
        result = save_file("../../etc/passwd", SIMPLE_CONTENT)
        assert "Ошибка валидации" in result


class TestNetworkErrors:
    def test_connect_error_returns_message(self, monkeypatch):
        def raise_connect_error(*args, **kwargs):
            raise httpx.ConnectError("Connection refused")

        monkeypatch.setattr(save_server.httpx, "post", raise_connect_error)
        result = save_file("file.txt", SIMPLE_CONTENT)
        assert "Не удалось подключиться" in result
        assert "localhost:8884" in result

    def test_timeout_error_returns_message(self, monkeypatch):
        def raise_timeout(*args, **kwargs):
            raise httpx.TimeoutException("Timeout", request=MagicMock())

        monkeypatch.setattr(save_server.httpx, "post", raise_timeout)
        result = save_file("file.txt", SIMPLE_CONTENT)
        assert "Превышено время ожидания" in result
