from datetime import datetime, timezone
from unittest.mock import MagicMock

import httpx
import pytest

import scheduler_server
from main import tasks, ReminderStatus, ReminderTask
from scheduler_server import (
    create_reminder,
    get_reminder_status,
    list_reminders,
    cancel_reminder,
    complete_reminder,
)


def _make_task(description="Test", delay_seconds=60, status=ReminderStatus.pending):
    now = datetime.now(timezone.utc)
    from datetime import timedelta
    import uuid
    task = ReminderTask(
        id=str(uuid.uuid4()),
        description=description,
        delay_seconds=delay_seconds,
        status=status,
        created_at=now,
        scheduled_at=now + timedelta(seconds=delay_seconds),
    )
    tasks[task.id] = task
    return task


class TestCreateReminder:
    def test_happy_path_format(self, valid_auth):
        result = create_reminder("Позвонить врачу", 60)
        assert "Напоминание" in result
        assert "Описание: Позвонить врачу" in result
        assert "Статус: pending" in result
        assert "Задержка: 60 сек." in result
        assert "Создано:" in result
        assert "Сработает в:" in result

    def test_delay_zero_returns_validation_error(self, valid_auth):
        result = create_reminder("Test", 0)
        assert "Ошибка валидации" in result
        assert "delay_seconds" in result

    def test_invalid_auth(self, invalid_auth):
        result = create_reminder("Test", 10)
        assert "Ошибка авторизации" in result
        assert "X-API-Key" in result


class TestGetReminderStatus:
    def test_happy_path(self, valid_auth):
        task = _make_task("Встреча", 120)
        result = get_reminder_status(task.id)
        assert f"Напоминание {task.id}:" in result
        assert "Описание: Встреча" in result
        assert "Статус: pending" in result
        assert "Задержка: 120 сек." in result

    def test_unknown_id_returns_not_found(self, valid_auth):
        result = get_reminder_status("non-existent-id")
        assert "не найдена" in result.lower() or "not found" in result.lower()

    def test_invalid_auth(self, invalid_auth):
        result = get_reminder_status("any-id")
        assert "Ошибка авторизации" in result
        assert "X-API-Key" in result


class TestListReminders:
    def test_empty_list(self, valid_auth):
        result = list_reminders()
        assert "Напоминаний нет" in result

    def test_with_tasks(self, valid_auth):
        _make_task("Задача 1", 30)
        _make_task("Задача 2", 60)
        result = list_reminders()
        assert "Напоминания (2):" in result
        assert "Задача 1" in result
        assert "Задача 2" in result

    def test_filter_status_pending(self, valid_auth):
        task = _make_task("Pending task", 60, status=ReminderStatus.pending)
        result = list_reminders(status="pending")
        assert task.id in result
        assert "pending" in result

    def test_filter_status_fired_empty(self, valid_auth):
        _make_task("Pending task", 60, status=ReminderStatus.pending)
        result = list_reminders(status="fired")
        assert "fired" in result
        assert "нет" in result.lower()

    def test_invalid_auth(self, invalid_auth):
        result = list_reminders()
        assert "Ошибка авторизации" in result
        assert "X-API-Key" in result


class TestCancelReminder:
    def test_pending_to_cancelled(self, valid_auth):
        task = _make_task("Отменить меня", 300)
        result = cancel_reminder(task.id)
        assert "успешно отменена" in result
        assert task.id in result

    def test_cancel_twice_returns_conflict(self, valid_auth):
        task = _make_task("Отменить меня дважды", 300)
        cancel_reminder(task.id)
        result = cancel_reminder(task.id)
        assert "Конфликт" in result or "Нельзя отменить" in result or "409" in result or "статус" in result

    def test_not_found(self, valid_auth):
        result = cancel_reminder("non-existent-id")
        assert "не найдена" in result.lower()

    def test_invalid_auth(self, invalid_auth):
        result = cancel_reminder("any-id")
        assert "Ошибка авторизации" in result
        assert "X-API-Key" in result


class TestCompleteReminder:
    def test_fired_to_completed(self, valid_auth):
        task = _make_task("Завершить меня", 1, status=ReminderStatus.fired)
        result = complete_reminder(task.id)
        assert "успешно завершена" in result
        assert task.id in result

    def test_conflict_on_pending(self, valid_auth):
        task = _make_task("Нельзя завершить pending", 60, status=ReminderStatus.pending)
        result = complete_reminder(task.id)
        assert "Конфликт" in result or "Нельзя завершить" in result or "статус" in result

    def test_not_found(self, valid_auth):
        result = complete_reminder("non-existent-id")
        assert "не найдена" in result.lower()

    def test_invalid_auth(self, invalid_auth):
        result = complete_reminder("any-id")
        assert "Ошибка авторизации" in result
        assert "X-API-Key" in result


class TestNetworkErrors:
    def test_create_connect_error(self, monkeypatch):
        def raise_connect_error(*args, **kwargs):
            raise httpx.ConnectError("Connection refused")

        monkeypatch.setattr(scheduler_server.httpx, "post", raise_connect_error)
        result = create_reminder("Test", 10)
        assert "Не удалось подключиться" in result
        assert "localhost:8001" in result

    def test_create_timeout_error(self, monkeypatch):
        def raise_timeout(*args, **kwargs):
            raise httpx.TimeoutException("Timeout", request=MagicMock())

        monkeypatch.setattr(scheduler_server.httpx, "post", raise_timeout)
        result = create_reminder("Test", 10)
        assert "Превышено время ожидания" in result

    def test_get_connect_error(self, monkeypatch):
        def raise_connect_error(*args, **kwargs):
            raise httpx.ConnectError("Connection refused")

        monkeypatch.setattr(scheduler_server.httpx, "get", raise_connect_error)
        result = get_reminder_status("any-id")
        assert "Не удалось подключиться" in result

    def test_get_timeout_error(self, monkeypatch):
        def raise_timeout(*args, **kwargs):
            raise httpx.TimeoutException("Timeout", request=MagicMock())

        monkeypatch.setattr(scheduler_server.httpx, "get", raise_timeout)
        result = get_reminder_status("any-id")
        assert "Превышено время ожидания" in result

    def test_list_connect_error(self, monkeypatch):
        def raise_connect_error(*args, **kwargs):
            raise httpx.ConnectError("Connection refused")

        monkeypatch.setattr(scheduler_server.httpx, "get", raise_connect_error)
        result = list_reminders()
        assert "Не удалось подключиться" in result

    def test_delete_connect_error(self, monkeypatch):
        def raise_connect_error(*args, **kwargs):
            raise httpx.ConnectError("Connection refused")

        monkeypatch.setattr(scheduler_server.httpx, "delete", raise_connect_error)
        result = cancel_reminder("any-id")
        assert "Не удалось подключиться" in result

    def test_delete_timeout_error(self, monkeypatch):
        def raise_timeout(*args, **kwargs):
            raise httpx.TimeoutException("Timeout", request=MagicMock())

        monkeypatch.setattr(scheduler_server.httpx, "delete", raise_timeout)
        result = cancel_reminder("any-id")
        assert "Превышено время ожидания" in result

    def test_patch_connect_error(self, monkeypatch):
        def raise_connect_error(*args, **kwargs):
            raise httpx.ConnectError("Connection refused")

        monkeypatch.setattr(scheduler_server.httpx, "patch", raise_connect_error)
        result = complete_reminder("any-id")
        assert "Не удалось подключиться" in result

    def test_patch_timeout_error(self, monkeypatch):
        def raise_timeout(*args, **kwargs):
            raise httpx.TimeoutException("Timeout", request=MagicMock())

        monkeypatch.setattr(scheduler_server.httpx, "patch", raise_timeout)
        result = complete_reminder("any-id")
        assert "Превышено время ожидания" in result
