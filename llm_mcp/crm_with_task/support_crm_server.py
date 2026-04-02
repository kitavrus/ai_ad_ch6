"""MCP-сервер для поддержки пользователей: пользователи и тикеты."""

import json
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

DATA_FILE = Path(__file__).parent / "support_data.json"
VALID_STATUSES = {"OPEN", "IN_PROGRESS", "RESOLVED", "CLOSED"}
VALID_CATEGORIES = {"auth", "billing", "api", "general"}

mcp = FastMCP("support-crm")


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _seed() -> dict:
    def ts(days_ago: int = 0, hours_ago: int = 0) -> str:
        return (datetime.now(timezone.utc) - timedelta(days=days_ago, hours=hours_ago)).isoformat()

    u1, u2, u3 = str(uuid.uuid4()), str(uuid.uuid4()), str(uuid.uuid4())
    t1, t2, t3, t4, t5 = [str(uuid.uuid4()) for _ in range(5)]

    users = {
        u1: {"id": u1, "name": "Алексей Иванов", "email": "ivanov@example.com", "plan": "pro", "created_at": ts(30)},
        u2: {"id": u2, "name": "Мария Петрова", "email": "petrova@example.com", "plan": "free", "created_at": ts(10)},
        u3: {"id": u3, "name": "Дмитрий Сидоров", "email": "sidorov@example.com", "plan": "enterprise", "created_at": ts(60)},
    }

    tickets = {
        t1: {
            "id": t1, "user_id": u1, "title": "Не работает авторизация через OAuth",
            "description": "При попытке войти через Google получаю ошибку 401. Redirect URI настроен верно.",
            "status": "OPEN", "category": "auth", "created_at": ts(hours_ago=2),
            "messages": [
                {"role": "user", "text": "Пробовал сбросить токен — не помогает.", "ts": ts(hours_ago=1)},
            ],
        },
        t2: {
            "id": t2, "user_id": u1, "title": "Вопрос по лимитам API на тарифе Pro",
            "description": "Сколько запросов в минуту доступно на Pro-плане?",
            "status": "RESOLVED", "category": "api", "created_at": ts(days_ago=3),
            "messages": [
                {"role": "support", "text": "На Pro-плане лимит 1000 req/min. Подробнее в документации.", "ts": ts(days_ago=2)},
            ],
        },
        t3: {
            "id": t3, "user_id": u2, "title": "Не могу сбросить пароль",
            "description": "Письмо для сброса пароля не приходит уже 30 минут.",
            "status": "IN_PROGRESS", "category": "auth", "created_at": ts(hours_ago=5),
            "messages": [],
        },
        t4: {
            "id": t4, "user_id": u2, "title": "Ошибка при оплате подписки",
            "description": "Карта отклонена, хотя на счёте достаточно средств. Код ошибки: CARD_DECLINED.",
            "status": "OPEN", "category": "billing", "created_at": ts(hours_ago=1),
            "messages": [],
        },
        t5: {
            "id": t5, "user_id": u3, "title": "Webhook не получает события",
            "description": "Настроил webhook URL, но события order.created не приходят. Endpoint возвращает 200.",
            "status": "OPEN", "category": "api", "created_at": ts(days_ago=1),
            "messages": [
                {"role": "user", "text": "Проверил firewall — порт открыт.", "ts": ts(hours_ago=20)},
            ],
        },
    }

    return {"users": users, "tickets": tickets}


def _load() -> dict:
    if not DATA_FILE.exists():
        data = _seed()
        _save(data)
        return data
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _save(data: dict) -> None:
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def list_users() -> list:
    """Return all users (for user selection dropdown).

    Returns:
        List of user objects with id, name, email, plan
    """
    data = _load()
    return list(data["users"].values())


@mcp.tool()
def get_user(user_id: str) -> dict:
    """Get a user by ID.

    Args:
        user_id: The UUID of the user

    Returns:
        User object or error dict
    """
    data = _load()
    if user_id not in data["users"]:
        return {"error": f"User '{user_id}' not found"}
    return data["users"][user_id]


@mcp.tool()
def list_tickets(user_id: Optional[str] = None, status: Optional[str] = None) -> list:
    """Return support tickets, optionally filtered by user and/or status.

    Args:
        user_id: Filter by user UUID (optional)
        status: Filter by status — OPEN, IN_PROGRESS, RESOLVED, CLOSED (optional)

    Returns:
        List of ticket objects sorted by created_at descending
    """
    if status is not None and status not in VALID_STATUSES:
        return [{"error": f"Invalid status '{status}'. Must be one of: {', '.join(sorted(VALID_STATUSES))}"}]

    data = _load()
    tickets = list(data["tickets"].values())

    if user_id is not None:
        tickets = [t for t in tickets if t["user_id"] == user_id]
    if status is not None:
        tickets = [t for t in tickets if t["status"] == status]

    tickets.sort(key=lambda t: t.get("created_at", ""), reverse=True)
    return tickets


@mcp.tool()
def get_ticket(ticket_id: str) -> dict:
    """Get a support ticket by ID, including its message history.

    Args:
        ticket_id: The UUID of the ticket

    Returns:
        Ticket object with messages or error dict
    """
    data = _load()
    if ticket_id not in data["tickets"]:
        return {"error": f"Ticket '{ticket_id}' not found"}
    return data["tickets"][ticket_id]


@mcp.tool()
def create_ticket(
    user_id: str,
    title: str,
    description: Optional[str] = None,
    category: str = "general",
) -> dict:
    """Create a new support ticket for a user.

    Args:
        user_id: The UUID of the user creating the ticket
        title: Short summary of the issue (required)
        description: Detailed description of the problem
        category: Issue category — auth, billing, api, general (default: general)

    Returns:
        The created ticket object
    """
    if category not in VALID_CATEGORIES:
        return {"error": f"Invalid category '{category}'. Must be one of: {', '.join(sorted(VALID_CATEGORIES))}"}

    data = _load()
    if user_id not in data["users"]:
        return {"error": f"User '{user_id}' not found"}

    ticket_id = str(uuid.uuid4())
    ticket = {
        "id": ticket_id,
        "user_id": user_id,
        "title": title,
        "description": description,
        "status": "OPEN",
        "category": category,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "messages": [],
    }
    data["tickets"][ticket_id] = ticket
    _save(data)
    return ticket


if __name__ == "__main__":
    mcp.run()
