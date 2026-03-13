import os
from typing import Optional

import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

mcp = FastMCP("save-to-file")

BASE_URL = "http://localhost:8884"
API_KEY = os.getenv("SAVE_API_KEY", "secret-token")
_HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}


@mcp.tool()
def save_file(filename: str, content_base64: str, subfolder: Optional[str] = None) -> str:
    """Сохраняет файл на диск через сервис save-to-file API.

    filename — имя файла (без пути)
    content_base64 — содержимое файла в base64
    subfolder — необязательная подпапка внутри SAVE_DIR

    Возвращает путь к сохранённому файлу или сообщение об ошибке.
    """
    try:
        payload = {"filename": filename, "content_base64": content_base64}
        if subfolder is not None:
            payload["subfolder"] = subfolder

        response = httpx.post(
            f"{BASE_URL}/save",
            headers=_HEADERS,
            json=payload,
            timeout=30.0,
        )

        if response.status_code == 200:
            data = response.json()
            saved_path = data["saved_path"]
            size_bytes = data["size_bytes"]
            return f"Файл сохранён: {saved_path} ({size_bytes} байт)"
        if response.status_code == 401:
            return "Ошибка авторизации: неверный или отсутствующий API ключ (X-API-Key)."
        if response.status_code == 422:
            detail = response.json().get("detail", "Ошибка валидации данных.")
            return f"Ошибка валидации: {detail}"
        if response.status_code == 400:
            detail = response.json().get("detail", "Недопустимый запрос.")
            return f"Ошибка: {detail}"
        detail = response.json().get("detail", "Внутренняя ошибка сервера.")
        return f"Ошибка сервера: {detail}"

    except httpx.ConnectError:
        return "Не удалось подключиться к сервису сохранения файлов. Убедитесь, что API запущен на localhost:8884."
    except httpx.TimeoutException:
        return "Превышено время ожидания ответа от сервиса сохранения файлов."


if __name__ == "__main__":
    mcp.run()
