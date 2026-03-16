import os

import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

mcp = FastMCP("pdf-maker")

BASE_URL = "http://localhost:8883"
SAVE_URL = "http://localhost:8884"
API_KEY = os.getenv("PDF_API_KEY", "secret-token")
SAVE_API_KEY = os.getenv("SAVE_API_KEY", "secret-token")
_HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}
_SAVE_HEADERS = {"X-API-Key": SAVE_API_KEY} if SAVE_API_KEY else {}


@mcp.tool()
def create_pdf(
    title: str,
    content: str | None = None,
    sections: list | None = None,
    author: str | None = None,
    file_path: str | None = None,
) -> str:
    """Создаёт PDF-документ и автоматически сохраняет его на диск.

    title   — заголовок документа, например "Отчёт за март"
    content — простой текстовый контент (необязательно). Если указан, будет добавлен
              как секция без заголовка. Удобно для быстрого создания PDF из текста.
    author  — автор документа (необязательно), например "Иван Иванов"
    sections — список секций (необязательно). Каждая секция — dict с полями:
        - heading: str          — заголовок секции
        - content: str          — текстовый параграф
        - items: list[str]      — маркированный список строк
        - table: {"headers": [...], "rows": [[...]]}  — таблица
    file_path — игнорируется (путь сохранения управляется сервером автоматически)

    Пример:
        create_pdf(title="Отчёт", content="Текст отчёта...")
        create_pdf(
            title="Мой документ",
            author="Автор",
            sections=[{"heading": "Введение", "content": "Текст..."}],
        )

    Возвращает путь к сохранённому файлу на диске.
    """
    try:
        # Build sections list: prepend plain content if provided
        built_sections = list(sections or [])
        if content:
            built_sections.insert(0, {"content": content})
        payload = {"title": title, "sections": built_sections, "author": author}

        response = httpx.post(
            f"{BASE_URL}/pdf",
            headers=_HEADERS,
            json=payload,
            timeout=30.0,
        )

        if response.status_code == 401:
            return "Ошибка авторизации: неверный или отсутствующий API ключ (X-API-Key)."
        if response.status_code == 422:
            detail = response.json().get("detail", "Ошибка валидации данных.")
            return f"Ошибка валидации: {detail}"
        if response.status_code != 200:
            return f"Ошибка API: статус {response.status_code}"

        data = response.json()
        filename = data["filename"]
        size = data["size_bytes"]
        pdf_b64 = data["pdf_base64"]

        # Автоматически сохраняем файл на диск
        try:
            save_response = httpx.post(
                f"{SAVE_URL}/save",
                headers=_SAVE_HEADERS,
                json={"filename": filename, "content_base64": pdf_b64},
                timeout=30.0,
            )
            if save_response.status_code == 200:
                saved_path = save_response.json()["saved_path"]
                return f"PDF создан и сохранён: {saved_path} ({size} байт)"
            save_detail = save_response.json().get("detail", f"статус {save_response.status_code}")
            return f"PDF создан ({size} байт), но не сохранён: {save_detail}"
        except httpx.ConnectError:
            return f"PDF создан ({size} байт), но не сохранён: сервис save-to-file недоступен на localhost:8884."

    except httpx.ConnectError:
        return "Не удалось подключиться к сервису PDF Maker. Убедитесь, что API запущен на localhost:8883."
    except httpx.TimeoutException:
        return "Превышено время ожидания ответа от сервиса PDF Maker."


if __name__ == "__main__":
    mcp.run()
