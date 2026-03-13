import asyncio
import os
import uvicorn
import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Optional

import httpx
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

load_dotenv()

app = FastAPI(title="Scheduler API", description="Reminder scheduling service")

API_KEY = os.getenv("SCHEDULER_API_KEY", "secret-token")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

tasks: dict[str, "ReminderTask"] = {}


async def verify_token(key: str = Depends(api_key_header)):
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Неверный или отсутствующий токен")


class ReminderStatus(str, Enum):
    pending = "pending"
    fired = "fired"
    completed = "completed"
    cancelled = "cancelled"
    failed = "failed"


class ReminderTask(BaseModel):
    id: str
    description: str
    delay_seconds: int
    webhook_url: Optional[str] = None
    status: ReminderStatus = ReminderStatus.pending
    created_at: datetime
    scheduled_at: datetime
    fired_at: Optional[datetime] = None
    webhook_response: Optional[str] = None


class CreateReminderRequest(BaseModel):
    description: str
    delay_seconds: int = Field(ge=1)
    webhook_url: Optional[str] = None


async def _fire_reminder(task_id: str) -> None:
    await asyncio.sleep(tasks[task_id].delay_seconds)

    task = tasks.get(task_id)
    if task is None or task.status == ReminderStatus.cancelled:
        return

    task.status = ReminderStatus.fired
    task.fired_at = datetime.now(timezone.utc)

    if task.webhook_url:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(task.webhook_url, json=task.model_dump(mode="json"))
            task.webhook_response = f"{resp.status_code} {resp.reason_phrase}"
            task.status = ReminderStatus.completed if resp.is_success else ReminderStatus.failed
        except Exception as exc:
            task.webhook_response = f"ERROR: {exc}"
            task.status = ReminderStatus.failed


@app.get("/")
def root():
    return {
        "service": "Scheduler API",
        "tasks_count": len(tasks),
        "endpoints": ["/reminders", "/reminders/{task_id}"],
    }


@app.post("/reminders", status_code=201, response_model=ReminderTask, dependencies=[Depends(verify_token)])
async def create_reminder(body: CreateReminderRequest):
    now = datetime.now(timezone.utc)
    task = ReminderTask(
        id=str(uuid.uuid4()),
        description=body.description,
        delay_seconds=body.delay_seconds,
        webhook_url=body.webhook_url,
        created_at=now,
        scheduled_at=now + timedelta(seconds=body.delay_seconds),
    )
    tasks[task.id] = task
    asyncio.create_task(_fire_reminder(task.id))
    return task


@app.get("/reminders", response_model=list[ReminderTask], dependencies=[Depends(verify_token)])
def list_reminders(status: Optional[ReminderStatus] = Query(default=None)):
    result = list(tasks.values())
    if status is not None:
        result = [t for t in result if t.status == status]
    return result


@app.get("/reminders/{task_id}", response_model=ReminderTask, dependencies=[Depends(verify_token)])
def get_reminder(task_id: str):
    task = tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Задача '{task_id}' не найдена")
    return task


@app.patch("/reminders/{task_id}/complete", response_model=ReminderTask, dependencies=[Depends(verify_token)])
def complete_reminder(task_id: str):
    task = tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Задача '{task_id}' не найдена")
    if task.status != ReminderStatus.fired:
        raise HTTPException(status_code=409, detail=f"Нельзя завершить задачу со статусом '{task.status}'")
    task.status = ReminderStatus.completed
    return task


@app.delete("/reminders/{task_id}", response_model=ReminderTask, dependencies=[Depends(verify_token)])
def cancel_reminder(task_id: str):
    task = tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Задача '{task_id}' не найдена")
    if task.status != ReminderStatus.pending:
        raise HTTPException(status_code=409, detail=f"Нельзя отменить задачу со статусом '{task.status}'")
    task.status = ReminderStatus.cancelled
    return task


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8881)
