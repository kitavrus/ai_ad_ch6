import os
import httpx
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

load_dotenv()

mcp = FastMCP("weather")

BASE_URL = "http://localhost:8882"
API_KEY = os.getenv("WEATHER_API_KEY", "")

_HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}


@mcp.tool()
def get_weather(city: str) -> str:
    """Возвращает текущую погоду для города. Используйте list_cities() для списка доступных городов."""
    try:
        response = httpx.get(f"{BASE_URL}/weather/{city}", headers=_HEADERS, timeout=10.0)
        if response.status_code == 401:
            return "Ошибка авторизации: неверный или отсутствующий API ключ (X-API-Key)."
        if response.status_code == 200:
            data = response.json()
            return (
                f"Погода в {data['city']}:\n"
                f"  Температура: {data['temperature']}°C\n"
                f"  Состояние: {data['condition']}\n"
                f"  Влажность: {data['humidity']}%\n"
                f"  Ветер: {data['wind_speed']} м/с, {data['wind_direction']}"
            )
        elif response.status_code == 404:
            return response.json().get("detail", f"Город '{city}' не найден.")
        else:
            return f"Ошибка API: статус {response.status_code}"
    except httpx.ConnectError:
        return "Не удалось подключиться к сервису погоды. Убедитесь, что API запущен на localhost:8000."
    except httpx.TimeoutException:
        return "Превышено время ожидания ответа от сервиса погоды."


@mcp.tool()
def list_cities() -> str:
    """Возвращает список всех доступных городов для запроса погоды."""
    try:
        response = httpx.get(f"{BASE_URL}/cities", headers=_HEADERS, timeout=10.0)
        if response.status_code == 401:
            return "Ошибка авторизации: неверный или отсутствующий API ключ (X-API-Key)."
        if response.status_code == 200:
            cities = response.json()
            return "Доступные города:\n" + "\n".join(f"  - {c}" for c in cities)
        else:
            return f"Ошибка API: статус {response.status_code}"
    except httpx.ConnectError:
        return "Не удалось подключиться к сервису погоды. Убедитесь, что API запущен на localhost:8000."
    except httpx.TimeoutException:
        return "Превышено время ожидания ответа от сервиса погоды."


if __name__ == "__main__":
    mcp.run()
