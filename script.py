from openai import OpenAI

# === НАСТРОЙКИ ===
API_KEY = ""
BASE_URL = "https://routerai.ru/api/v1"

# Модель
#MODEL = "openai/gpt-4o"
MODEL = "deepseek/deepseek-v3.2"

# Системный промпт (если не нужен — оставьте пустой строкой)
SYSTEM_PROMPT = "Ты полезный ассистент."

# Основной промпт пользователя
USER_PROMPT = "Привет! продолжи текст после слова стоп1, повтори что я тут написал"

# Модификатор, добавляемый к промпту пользователя
PROMPT_MODIFIER = "Ответь кратко. Да или Нет"

# Явное описание формата ответа (будет добавлено к системному промпту или отдельным сообщением)
FORMAT_DESCRIPTION = "Ответ должен быть в формате JSON с полями 'answer' и 'confidence'."

# Инструкция по завершению ответа (например, "Закончи ответ словом 'КОНЕЦ'.")
COMPLETION_INSTRUCTION = ""

# Стоп-последовательности (через запятую). Если не нужны — оставьте пустым.
STOP_SEQUENCES = "стоп1, стоп2, стоп3"

# Ограничение на длину ответа (в токенах)
MAX_TOKENS = 150

# Температура (0-2)
TEMPERATURE = 0.7

# Top_p (0-1)
TOP_P = 0.9

# Top_k (целое число, поддерживается не всеми API)
TOP_K = 50

# === ПОДГОТОВКА ДАННЫХ ===
# Разбираем стоп-последовательности в список
stop_list = [s.strip() for s in STOP_SEQUENCES.split(",")] if STOP_SEQUENCES else None

# Формируем сообщения
messages = []

# Добавляем системный промпт (если есть)
system_content = SYSTEM_PROMPT
if FORMAT_DESCRIPTION:
    if system_content:
        system_content += "\n\n" + FORMAT_DESCRIPTION
    else:
        system_content = FORMAT_DESCRIPTION
if system_content:
    messages.append({"role": "system", "content": system_content})

# Добавляем инструкцию по завершению в промпт пользователя, если она задана
user_content = USER_PROMPT
if PROMPT_MODIFIER:
    user_content += PROMPT_MODIFIER
if COMPLETION_INSTRUCTION:
    user_content += "\n\n" + COMPLETION_INSTRUCTION

messages.append({"role": "user", "content": user_content})

# === СОЗДАНИЕ КЛИЕНТА ===
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# === ВЫПОЛНЕНИЕ ЗАПРОСА ===
extra_params = {}
if TOP_K is not None:
    extra_params["top_k"] = TOP_K

response = client.chat.completions.create(
    model=MODEL,
    messages=messages,
    max_tokens=MAX_TOKENS,
    temperature=TEMPERATURE,
    top_p=TOP_P,
    stop=stop_list,
    extra_body=extra_params if extra_params else None
)

# === ВЫВОД РЕЗУЛЬТАТА ===
print(response.choices[0].message.content)