from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from main import ReminderStatus, _fire_reminder, tasks


class TestRootEndpoint:
    def test_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_no_auth_required(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_response_structure(self, client):
        data = client.get("/").json()
        assert "service" in data
        assert "tasks_count" in data
        assert "endpoints" in data

    def test_service_name(self, client):
        data = client.get("/").json()
        assert data["service"] == "Scheduler API"

    def test_tasks_count_starts_at_zero(self, client):
        data = client.get("/").json()
        assert data["tasks_count"] == 0


class TestAuth:
    PROTECTED = [
        ("GET", "/reminders"),
        ("POST", "/reminders"),
        ("GET", "/reminders/nonexistent"),
        ("PATCH", "/reminders/nonexistent/complete"),
        ("DELETE", "/reminders/nonexistent"),
    ]

    def test_missing_key_post_reminders(self, client, sample_payload):
        response = client.post("/reminders", json=sample_payload)
        assert response.status_code == 401

    def test_wrong_key_post_reminders(self, client, invalid_headers, sample_payload):
        response = client.post("/reminders", json=sample_payload, headers=invalid_headers)
        assert response.status_code == 401

    def test_missing_key_get_reminders(self, client):
        response = client.get("/reminders")
        assert response.status_code == 401

    def test_wrong_key_get_reminders(self, client, invalid_headers):
        response = client.get("/reminders", headers=invalid_headers)
        assert response.status_code == 401

    def test_missing_key_get_by_id(self, client):
        response = client.get("/reminders/some-id")
        assert response.status_code == 401

    def test_wrong_key_get_by_id(self, client, invalid_headers):
        response = client.get("/reminders/some-id", headers=invalid_headers)
        assert response.status_code == 401

    def test_missing_key_patch_complete(self, client):
        response = client.patch("/reminders/some-id/complete")
        assert response.status_code == 401

    def test_missing_key_delete(self, client):
        response = client.delete("/reminders/some-id")
        assert response.status_code == 401

    def test_401_detail_message(self, client, invalid_headers):
        response = client.get("/reminders", headers=invalid_headers)
        assert response.json()["detail"] == "Неверный или отсутствующий токен"


class TestCreateReminder:
    def test_returns_201(self, client, valid_headers, sample_payload):
        response = client.post("/reminders", json=sample_payload, headers=valid_headers)
        assert response.status_code == 201

    def test_response_has_id(self, client, valid_headers, sample_payload):
        data = client.post("/reminders", json=sample_payload, headers=valid_headers).json()
        assert "id" in data
        assert data["id"]

    def test_response_has_description(self, client, valid_headers, sample_payload):
        data = client.post("/reminders", json=sample_payload, headers=valid_headers).json()
        assert data["description"] == sample_payload["description"]

    def test_response_has_status_pending(self, client, valid_headers, sample_payload):
        data = client.post("/reminders", json=sample_payload, headers=valid_headers).json()
        assert data["status"] == "pending"

    def test_response_has_created_at(self, client, valid_headers, sample_payload):
        data = client.post("/reminders", json=sample_payload, headers=valid_headers).json()
        assert "created_at" in data

    def test_response_has_scheduled_at(self, client, valid_headers, sample_payload):
        data = client.post("/reminders", json=sample_payload, headers=valid_headers).json()
        assert "scheduled_at" in data

    def test_delay_zero_returns_422(self, client, valid_headers):
        response = client.post("/reminders", json={"description": "test", "delay_seconds": 0}, headers=valid_headers)
        assert response.status_code == 422

    def test_delay_negative_returns_422(self, client, valid_headers):
        response = client.post("/reminders", json={"description": "test", "delay_seconds": -5}, headers=valid_headers)
        assert response.status_code == 422

    def test_two_reminders_have_different_ids(self, client, valid_headers, sample_payload):
        id1 = client.post("/reminders", json=sample_payload, headers=valid_headers).json()["id"]
        id2 = client.post("/reminders", json=sample_payload, headers=valid_headers).json()["id"]
        assert id1 != id2

    def test_webhook_url_stored(self, client, valid_headers, sample_payload_with_webhook):
        data = client.post("/reminders", json=sample_payload_with_webhook, headers=valid_headers).json()
        assert data["webhook_url"] == sample_payload_with_webhook["webhook_url"]

    def test_without_webhook_url_is_none(self, client, valid_headers, sample_payload):
        data = client.post("/reminders", json=sample_payload, headers=valid_headers).json()
        assert data["webhook_url"] is None


class TestGetReminder:
    def test_returns_200_for_existing(self, client, valid_headers, sample_payload):
        created = client.post("/reminders", json=sample_payload, headers=valid_headers).json()
        response = client.get(f"/reminders/{created['id']}", headers=valid_headers)
        assert response.status_code == 200

    def test_returns_404_for_unknown(self, client, valid_headers):
        response = client.get("/reminders/nonexistent-id", headers=valid_headers)
        assert response.status_code == 404

    def test_fields_match_creation(self, client, valid_headers, sample_payload):
        created = client.post("/reminders", json=sample_payload, headers=valid_headers).json()
        fetched = client.get(f"/reminders/{created['id']}", headers=valid_headers).json()
        assert fetched["id"] == created["id"]
        assert fetched["description"] == created["description"]
        assert fetched["delay_seconds"] == created["delay_seconds"]
        assert fetched["status"] == created["status"]


class TestListReminders:
    def test_empty_list_initially(self, client, valid_headers):
        response = client.get("/reminders", headers=valid_headers)
        assert response.status_code == 200
        assert response.json() == []

    def test_returns_created_reminder(self, client, valid_headers, sample_payload):
        client.post("/reminders", json=sample_payload, headers=valid_headers)
        data = client.get("/reminders", headers=valid_headers).json()
        assert len(data) == 1

    def test_filter_by_status_pending(self, client, valid_headers, sample_payload):
        client.post("/reminders", json=sample_payload, headers=valid_headers)
        data = client.get("/reminders?status=pending", headers=valid_headers).json()
        assert len(data) == 1
        assert all(t["status"] == "pending" for t in data)

    def test_filter_by_status_fired_empty(self, client, valid_headers, sample_payload):
        client.post("/reminders", json=sample_payload, headers=valid_headers)
        data = client.get("/reminders?status=fired", headers=valid_headers).json()
        assert data == []

    def test_invalid_status_returns_422(self, client, valid_headers):
        response = client.get("/reminders?status=invalid", headers=valid_headers)
        assert response.status_code == 422


class TestCancelReminder:
    def test_cancel_pending_returns_200(self, client, valid_headers, sample_payload):
        created = client.post("/reminders", json=sample_payload, headers=valid_headers).json()
        response = client.delete(f"/reminders/{created['id']}", headers=valid_headers)
        assert response.status_code == 200

    def test_cancel_pending_sets_cancelled(self, client, valid_headers, sample_payload):
        created = client.post("/reminders", json=sample_payload, headers=valid_headers).json()
        data = client.delete(f"/reminders/{created['id']}", headers=valid_headers).json()
        assert data["status"] == "cancelled"

    def test_cancel_unknown_returns_404(self, client, valid_headers):
        response = client.delete("/reminders/nonexistent", headers=valid_headers)
        assert response.status_code == 404

    def test_cancel_already_cancelled_returns_409(self, client, valid_headers, sample_payload):
        created = client.post("/reminders", json=sample_payload, headers=valid_headers).json()
        client.delete(f"/reminders/{created['id']}", headers=valid_headers)
        response = client.delete(f"/reminders/{created['id']}", headers=valid_headers)
        assert response.status_code == 409

    def test_cancel_completed_returns_409(self, client, valid_headers, sample_payload):
        # Manually set status to completed
        created = client.post("/reminders", json=sample_payload, headers=valid_headers).json()
        tasks[created["id"]].status = ReminderStatus.completed
        response = client.delete(f"/reminders/{created['id']}", headers=valid_headers)
        assert response.status_code == 409

    def test_cancel_fired_returns_409(self, client, valid_headers, sample_payload):
        created = client.post("/reminders", json=sample_payload, headers=valid_headers).json()
        tasks[created["id"]].status = ReminderStatus.fired
        response = client.delete(f"/reminders/{created['id']}", headers=valid_headers)
        assert response.status_code == 409


class TestCompleteReminder:
    def test_complete_fired_returns_200(self, client, valid_headers, sample_payload):
        created = client.post("/reminders", json=sample_payload, headers=valid_headers).json()
        tasks[created["id"]].status = ReminderStatus.fired
        response = client.patch(f"/reminders/{created['id']}/complete", headers=valid_headers)
        assert response.status_code == 200

    def test_complete_fired_sets_completed(self, client, valid_headers, sample_payload):
        created = client.post("/reminders", json=sample_payload, headers=valid_headers).json()
        tasks[created["id"]].status = ReminderStatus.fired
        data = client.patch(f"/reminders/{created['id']}/complete", headers=valid_headers).json()
        assert data["status"] == "completed"

    def test_complete_unknown_returns_404(self, client, valid_headers):
        response = client.patch("/reminders/nonexistent/complete", headers=valid_headers)
        assert response.status_code == 404

    def test_complete_pending_returns_409(self, client, valid_headers, sample_payload):
        created = client.post("/reminders", json=sample_payload, headers=valid_headers).json()
        response = client.patch(f"/reminders/{created['id']}/complete", headers=valid_headers)
        assert response.status_code == 409

    def test_complete_cancelled_returns_409(self, client, valid_headers, sample_payload):
        created = client.post("/reminders", json=sample_payload, headers=valid_headers).json()
        tasks[created["id"]].status = ReminderStatus.cancelled
        response = client.patch(f"/reminders/{created['id']}/complete", headers=valid_headers)
        assert response.status_code == 409


@pytest.mark.anyio
class TestBackgroundFiring:
    async def test_fire_sets_status_fired(self):
        from datetime import timedelta
        from main import ReminderTask, tasks

        now = __import__("datetime").datetime.now(__import__("datetime").timezone.utc)
        task = ReminderTask(
            id="test-fire-1",
            description="test",
            delay_seconds=1,
            created_at=now,
            scheduled_at=now + timedelta(seconds=1),
        )
        tasks[task.id] = task

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await _fire_reminder(task.id)

        assert tasks["test-fire-1"].status == ReminderStatus.fired

    async def test_fire_sets_fired_at(self):
        from datetime import timedelta
        from main import ReminderTask, tasks

        now = __import__("datetime").datetime.now(__import__("datetime").timezone.utc)
        task = ReminderTask(
            id="test-fire-2",
            description="test",
            delay_seconds=1,
            created_at=now,
            scheduled_at=now + timedelta(seconds=1),
        )
        tasks[task.id] = task

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await _fire_reminder(task.id)

        assert tasks["test-fire-2"].fired_at is not None

    async def test_fire_cancelled_task_no_op(self):
        from datetime import timedelta
        from main import ReminderTask, tasks

        now = __import__("datetime").datetime.now(__import__("datetime").timezone.utc)
        task = ReminderTask(
            id="test-fire-3",
            description="test",
            delay_seconds=1,
            created_at=now,
            scheduled_at=now + timedelta(seconds=1),
            status=ReminderStatus.cancelled,
        )
        tasks[task.id] = task

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await _fire_reminder(task.id)

        assert tasks["test-fire-3"].status == ReminderStatus.cancelled


@pytest.mark.anyio
class TestWebhookCalling:
    async def _make_task(self, webhook_url=None):
        from datetime import timedelta
        from main import ReminderTask, tasks

        now = __import__("datetime").datetime.now(__import__("datetime").timezone.utc)
        task = ReminderTask(
            id=str(__import__("uuid").uuid4()),
            description="webhook-test",
            delay_seconds=1,
            webhook_url=webhook_url,
            created_at=now,
            scheduled_at=now + timedelta(seconds=1),
        )
        tasks[task.id] = task
        return task

    async def test_webhook_200_sets_completed(self):
        task = await self._make_task(webhook_url="http://example.com/hook")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.reason_phrase = "OK"
        mock_resp.is_success = True

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch("asyncio.sleep", new_callable=AsyncMock), \
             patch("httpx.AsyncClient", return_value=mock_client):
            await _fire_reminder(task.id)

        assert tasks[task.id].status == ReminderStatus.completed
        assert "200" in tasks[task.id].webhook_response

    async def test_webhook_500_sets_failed(self):
        task = await self._make_task(webhook_url="http://example.com/hook")

        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.reason_phrase = "Internal Server Error"
        mock_resp.is_success = False

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch("asyncio.sleep", new_callable=AsyncMock), \
             patch("httpx.AsyncClient", return_value=mock_client):
            await _fire_reminder(task.id)

        assert tasks[task.id].status == ReminderStatus.failed

    async def test_webhook_exception_sets_failed(self):
        task = await self._make_task(webhook_url="http://example.com/hook")

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(side_effect=Exception("connection refused"))

        with patch("asyncio.sleep", new_callable=AsyncMock), \
             patch("httpx.AsyncClient", return_value=mock_client):
            await _fire_reminder(task.id)

        assert tasks[task.id].status == ReminderStatus.failed
        assert "ERROR" in tasks[task.id].webhook_response

    async def test_no_webhook_status_remains_fired(self):
        task = await self._make_task(webhook_url=None)

        with patch("asyncio.sleep", new_callable=AsyncMock), \
             patch("httpx.AsyncClient") as mock_httpx:
            await _fire_reminder(task.id)
            mock_httpx.assert_not_called()

        assert tasks[task.id].status == ReminderStatus.fired
