import os
import uvicorn
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import List

app = FastAPI(title="Weather API", description="Weather data for Russian cities")

API_KEY = os.getenv("WEATHER_API_KEY", "secret-token")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_token(key: str = Depends(api_key_header)):
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Неверный или отсутствующий токен")


class WeatherResponse(BaseModel):
    city: str
    temperature: int
    condition: str
    humidity: int
    wind_speed: int
    wind_direction: str


WEATHER_DATA = {
    "москва": WeatherResponse(city="Москва", temperature=-3, condition="Облачно, небольшой снег", humidity=80, wind_speed=5, wind_direction="С"),
    "санкт-петербург": WeatherResponse(city="Санкт-Петербург", temperature=-1, condition="Пасмурно, лёгкий снег", humidity=85, wind_speed=8, wind_direction="З"),
    "новосибирск": WeatherResponse(city="Новосибирск", temperature=-15, condition="Ясно, морозно", humidity=70, wind_speed=3, wind_direction="В"),
    "екатеринбург": WeatherResponse(city="Екатеринбург", temperature=-10, condition="Облачно", humidity=75, wind_speed=6, wind_direction="СЗ"),
    "казань": WeatherResponse(city="Казань", temperature=-8, condition="Снегопад", humidity=88, wind_speed=7, wind_direction="З"),
    "нижний новгород": WeatherResponse(city="Нижний Новгород", temperature=-6, condition="Пасмурно", humidity=82, wind_speed=4, wind_direction="СВ"),
    "челябинск": WeatherResponse(city="Челябинск", temperature=-12, condition="Ясно", humidity=68, wind_speed=5, wind_direction="В"),
    "самара": WeatherResponse(city="Самара", temperature=-7, condition="Облачно, гололёд", humidity=78, wind_speed=9, wind_direction="ЮЗ"),
    "уфа": WeatherResponse(city="Уфа", temperature=-11, condition="Снег", humidity=84, wind_speed=6, wind_direction="З"),
    "ростов-на-дону": WeatherResponse(city="Ростов-на-Дону", temperature=3, condition="Дождь", humidity=90, wind_speed=12, wind_direction="Ю"),
    "краснодар": WeatherResponse(city="Краснодар", temperature=5, condition="Переменная облачность", humidity=76, wind_speed=10, wind_direction="ЮЗ"),
    "пермь": WeatherResponse(city="Пермь", temperature=-14, condition="Ясно, морозно", humidity=72, wind_speed=4, wind_direction="С"),
    "воронеж": WeatherResponse(city="Воронеж", temperature=-5, condition="Облачно", humidity=81, wind_speed=7, wind_direction="СЗ"),
    "волгоград": WeatherResponse(city="Волгоград", temperature=-2, condition="Пасмурно", humidity=83, wind_speed=11, wind_direction="З"),
    "красноярск": WeatherResponse(city="Красноярск", temperature=-18, condition="Ясно, сильный мороз", humidity=65, wind_speed=2, wind_direction="В"),
    "саратов": WeatherResponse(city="Саратов", temperature=-4, condition="Лёгкий снег", humidity=79, wind_speed=8, wind_direction="ЮЗ"),
    "тюмень": WeatherResponse(city="Тюмень", temperature=-16, condition="Метель", humidity=87, wind_speed=13, wind_direction="СВ"),
    "тольятти": WeatherResponse(city="Тольятти", temperature=-8, condition="Облачно", humidity=77, wind_speed=5, wind_direction="З"),
    "ижевск": WeatherResponse(city="Ижевск", temperature=-13, condition="Снег", humidity=86, wind_speed=6, wind_direction="С"),
    "барнаул": WeatherResponse(city="Барнаул", temperature=-17, condition="Ясно", humidity=66, wind_speed=3, wind_direction="В"),
    "иркутск": WeatherResponse(city="Иркутск", temperature=-20, condition="Ясно, сильный мороз", humidity=62, wind_speed=2, wind_direction="ЮВ"),
    "ульяновск": WeatherResponse(city="Ульяновск", temperature=-9, condition="Пасмурно, снег", humidity=83, wind_speed=7, wind_direction="З"),
    "хабаровск": WeatherResponse(city="Хабаровск", temperature=-22, condition="Ясно, морозно", humidity=60, wind_speed=4, wind_direction="СЗ"),
    "ярославль": WeatherResponse(city="Ярославль", temperature=-7, condition="Облачно, снег", humidity=84, wind_speed=6, wind_direction="С"),
    "владивосток": WeatherResponse(city="Владивосток", temperature=-8, condition="Ясно", humidity=70, wind_speed=9, wind_direction="С"),
    "махачкала": WeatherResponse(city="Махачкала", temperature=6, condition="Переменная облачность", humidity=73, wind_speed=11, wind_direction="Ю"),
    "томск": WeatherResponse(city="Томск", temperature=-19, condition="Снег", humidity=88, wind_speed=5, wind_direction="В"),
    "оренбург": WeatherResponse(city="Оренбург", temperature=-13, condition="Ясно", humidity=69, wind_speed=8, wind_direction="ЮВ"),
    "кемерово": WeatherResponse(city="Кемерово", temperature=-16, condition="Облачно", humidity=74, wind_speed=4, wind_direction="В"),
    "омск": WeatherResponse(city="Омск", temperature=-14, condition="Ясно", humidity=67, wind_speed=6, wind_direction="СВ"),
    "новокузнецк": WeatherResponse(city="Новокузнецк", temperature=-17, condition="Снег", humidity=85, wind_speed=5, wind_direction="В"),
    "рязань": WeatherResponse(city="Рязань", temperature=-6, condition="Пасмурно", humidity=80, wind_speed=7, wind_direction="СЗ"),
    "астрахань": WeatherResponse(city="Астрахань", temperature=1, condition="Облачно, туман", humidity=91, wind_speed=10, wind_direction="Ю"),
    "пенза": WeatherResponse(city="Пенза", temperature=-8, condition="Снег", humidity=82, wind_speed=6, wind_direction="З"),
    "липецк": WeatherResponse(city="Липецк", temperature=-5, condition="Облачно", humidity=79, wind_speed=8, wind_direction="СЗ"),
}

ALIASES = {
    "spb": "санкт-петербург",
    "мск": "москва",
    "питер": "санкт-петербург",
    "нн": "нижний новгород",
    "ростов": "ростов-на-дону",
    # English aliases
    "moscow": "москва",
    "saint petersburg": "санкт-петербург",
    "saint-petersburg": "санкт-петербург",
    "st. petersburg": "санкт-петербург",
    "novosibirsk": "новосибирск",
    "yekaterinburg": "екатеринбург",
    "ekaterinburg": "екатеринбург",
    "kazan": "казань",
    "nizhny novgorod": "нижний новгород",
    "chelyabinsk": "челябинск",
    "samara": "самара",
    "ufa": "уфа",
    "rostov-on-don": "ростов-на-дону",
    "krasnodar": "краснодар",
    "perm": "пермь",
    "voronezh": "воронеж",
    "volgograd": "волгоград",
    "krasnoyarsk": "красноярск",
    "saratov": "саратов",
    "tyumen": "тюмень",
    "tolyatti": "тольятти",
    "togliatti": "тольятти",
    "izhevsk": "ижевск",
    "barnaul": "барнаул",
    "irkutsk": "иркутск",
    "ulyanovsk": "ульяновск",
    "khabarovsk": "хабаровск",
    "yaroslavl": "ярославль",
    "vladivostok": "владивосток",
    "makhachkala": "махачкала",
    "tomsk": "томск",
    "orenburg": "оренбург",
    "kemerovo": "кемерово",
    "omsk": "омск",
    "novokuznetsk": "новокузнецк",
    "ryazan": "рязань",
    "astrakhan": "астрахань",
    "penza": "пенза",
    "lipetsk": "липецк",
}


@app.get("/")
def root():
    return {"service": "Weather API", "cities_count": len(WEATHER_DATA), "endpoints": ["/weather/{city_name}", "/cities"]}


@app.get("/cities", response_model=List[str], dependencies=[Depends(verify_token)])
def list_cities():
    return [data.city for data in WEATHER_DATA.values()]


@app.get("/weather/{city_name}", response_model=WeatherResponse, dependencies=[Depends(verify_token)])
def get_weather(city_name: str):
    key = city_name.lower().strip()
    key = ALIASES.get(key, key)
    if key in WEATHER_DATA:
        return WEATHER_DATA[key]
    available = ", ".join(data.city for data in WEATHER_DATA.values())
    raise HTTPException(status_code=404, detail=f"Город '{city_name}' не найден. Доступные города: {available}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8882)
