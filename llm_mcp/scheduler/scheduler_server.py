import os
import httpx
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

load_dotenv()

mcp = FastMCP("scheduler")

BASE_URL = "http://localhost:8881"
API_KEY = os.getenv("SCHEDULER_API_KEY", "secret-token")
_HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}


def _format_reminder(data: dict) -> str:
    lines = [
        f"Напоминание {data['id']}:",
        f"  Описание: {data['description']}",
        f"  Статус: {data['status']}",
        f"  Задержка: {data['delay_seconds']} сек.",
        f"  Создано: {data['created_at']}",
        f"  Сработает в: {data['scheduled_at']}",
    ]
    if data.get("fired_at") is not None:
        lines.append(f"  Сработало: {data['fired_at']}")
    if data.get("webhook_response") is not None:
        lines.append(f"  Webhook: {data['webhook_response']}")
    return "\n".join(lines)


@mcp.tool()
def create_reminder(description: str, delay_seconds: int, webhook_url: str = None) -> str:
    """Создать напоминание с заданной задержкой в секундах."""
    try:
        response = httpx.post(
            f"{BASE_URL}/reminders",
            headers=_HEADERS,
            json={"description": description, "delay_seconds": delay_seconds, "webhook_url": webhook_url},
            timeout=10.0,
        )
        if response.status_code == 401:
            return "Ошибка авторизации: неверный или отсутствующий API ключ (X-API-Key)."
        if response.status_code == 201:
            return _format_reminder(response.json())
        if response.status_code == 422:
            return "Ошибка валидации: delay_seconds должен быть ≥ 1."
        return f"Ошибка API: статус {response.status_code}"
    except httpx.ConnectError:
        return "Не удалось подключиться к сервису планировщика. Убедитесь, что API запущен на localhost:8001."
    except httpx.TimeoutException:
        return "Превышено время ожидания ответа от сервиса планировщика."


@mcp.tool()
def get_reminder_status(task_id: str) -> str:
    """Получить статус напоминания по ID."""
    try:
        response = httpx.get(f"{BASE_URL}/reminders/{task_id}", headers=_HEADERS, timeout=10.0)
        if response.status_code == 401:
            return "Ошибка авторизации: неверный или отсутствующий API ключ (X-API-Key)."
        if response.status_code == 200:
            return _format_reminder(response.json())
        if response.status_code == 404:
            return response.json().get("detail", "Задача не найдена.")
        return f"Ошибка API: статус {response.status_code}"
    except httpx.ConnectError:
        return "Не удалось подключиться к сервису планировщика. Убедитесь, что API запущен на localhost:8001."
    except httpx.TimeoutException:
        return "Превышено время ожидания ответа от сервиса планировщика."


@mcp.tool()
def list_reminders(status: str = None) -> str:
    """Получить список напоминаний. Опционально фильтр по статусу: pending, fired, completed, cancelled, failed."""
    try:
        params = {"status": status} if status else {}
        response = httpx.get(f"{BASE_URL}/reminders", headers=_HEADERS, params=params, timeout=10.0)
        if response.status_code == 401:
            return "Ошибка авторизации: неверный или отсутствующий API ключ (X-API-Key)."
        if response.status_code == 200:
            items = response.json()
            if not items:
                if status:
                    return f"Напоминаний со статусом '{status}' нет."
                return "Напоминаний нет."
            lines = [f"Напоминания ({len(items)}):"]
            for item in items:
                lines.append(f"  [{item['id']}] {item['status']} — {item['description']}")
            return "\n".join(lines)
        return f"Ошибка API: статус {response.status_code}"
    except httpx.ConnectError:
        return "Не удалось подключиться к сервису планировщика. Убедитесь, что API запущен на localhost:8001."
    except httpx.TimeoutException:
        return "Превышено время ожидания ответа от сервиса планировщика."


@mcp.tool()
def cancel_reminder(task_id: str) -> str:
    """Отменить напоминание (только в статусе pending)."""
    try:
        response = httpx.delete(f"{BASE_URL}/reminders/{task_id}", headers=_HEADERS, timeout=10.0)
        if response.status_code == 401:
            return "Ошибка авторизации: неверный или отсутствующий API ключ (X-API-Key)."
        if response.status_code == 200:
            return f"Задача {task_id} успешно отменена."
        if response.status_code == 404:
            return response.json().get("detail", "Задача не найдена.")
        if response.status_code == 409:
            return response.json().get("detail", "Конфликт статуса задачи.")
        return f"Ошибка API: статус {response.status_code}"
    except httpx.ConnectError:
        return "Не удалось подключиться к сервису планировщика. Убедитесь, что API запущен на localhost:8001."
    except httpx.TimeoutException:
        return "Превышено время ожидания ответа от сервиса планировщика."


@mcp.tool()
def complete_reminder(task_id: str) -> str:
    """Завершить напоминание вручную (только в статусе fired)."""
    try:
        response = httpx.patch(f"{BASE_URL}/reminders/{task_id}/complete", headers=_HEADERS, timeout=10.0)
        if response.status_code == 401:
            return "Ошибка авторизации: неверный или отсутствующий API ключ (X-API-Key)."
        if response.status_code == 200:
            return f"Задача {task_id} успешно завершена."
        if response.status_code == 404:
            return response.json().get("detail", "Задача не найдена.")
        if response.status_code == 409:
            return response.json().get("detail", "Конфликт статуса задачи.")
        return f"Ошибка API: статус {response.status_code}"
    except httpx.ConnectError:
        return "Не удалось подключиться к сервису планировщика. Убедитесь, что API запущен на localhost:8001."
    except httpx.TimeoutException:
        return "Превышено время ожидания ответа от сервиса планировщика."


if __name__ == "__main__":
    mcp.run()
