from mcp.server.fastmcp import FastMCP

mcp = FastMCP("weather")

WEATHER_DATA = {
    "москва": {
        "city": "Москва",
        "temperature": -3,
        "condition": "Облачно с прояснениями",
        "humidity": 78,
        "wind_speed": 5,
        "wind_direction": "СЗ",
    },
    "санкт-петербург": {
        "city": "Санкт-Петербург",
        "temperature": -1,
        "condition": "Пасмурно, лёгкий снег",
        "humidity": 85,
        "wind_speed": 8,
        "wind_direction": "З",
    },
}

@mcp.tool()
def get_weather(city: str) -> str:
    """Возвращает текущую погоду для города. Поддерживаемые города: Москва, Санкт-Петербург."""
    key = city.lower().strip()
    # aliases
    if key in ("spb", "питер", "saint petersburg", "st. petersburg"):
        key = "санкт-петербург"
    elif key in ("moscow", "msk"):
        key = "москва"

    data = WEATHER_DATA.get(key)
    if not data:
        return f"Погода для '{city}' недоступна. Поддерживаемые города: Москва, Санкт-Петербург."

    return (
        f"Погода в {data['city']}:\n"
        f"  Температура: {data['temperature']}°C\n"
        f"  Состояние: {data['condition']}\n"
        f"  Влажность: {data['humidity']}%\n"
        f"  Ветер: {data['wind_speed']} м/с, {data['wind_direction']}"
    )

if __name__ == "__main__":
    mcp.run()
