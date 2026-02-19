from openai import OpenAI
import os

# === НАСТРОЙКИ ===
# Базовый URL API
BASE_URL = "https://routerai.ru/api/v1"
# API ключ для авторизации
API_KEY = os.getenv("API_KEY")

# === Параметры модели ===
# Модель для генерации (выбранная модель)
#MODEL = "deepseek/deepseek-v3.2"
MODEL = "qwen/qwen3.5-plus-02-15"
# Максимальное количество токенов в ответе
MAX_TOKENS = 150
# Температура генерации (уровень случайности, 0-2)
TEMPERATURE = 0.7
# Top-p: вероятность влияния на выбор следующего токена (0-1)
TOP_P = 0.9
# Top-k: ограничение по количеству кандидатов
TOP_K = 50

# === Параметры промптов ===
# Системный промпт, задающий контекст ассистента //Ты полезный ассистент.
SYSTEM_PROMPT = "Ты финансовый аналитик"
#SYSTEM_PROMPT = "Ты финансовый блогер"
#SYSTEM_PROMPT = "Ты финансовый астролог"
# Основной пользовательский промпт (начало беседы)
USER_PROMPT = "Спрогнозируй курс рубля к доллару на 2026 год"
# Модификатор к промпту, добавляющий инструкции к сообщению пользователя
PROMPT_MODIFIER = "Ответь кратко"
# Описание формата ответа, добавляемое к промпту (например, JSON с полями)
FORMAT_DESCRIPTION = "Ответ должен быть в dm формате"
# Инструкция по завершению ответа
COMPLETION_INSTRUCTION = ""
# Стоп-последовательности (массив строк), определяющие прекращение генерации //"стоп1", "стоп2", "стоп3"
STOP_SEQUENCES = []


def build_messages():
    """Формирует список сообщений для API."""
    messages = []

    system_content = SYSTEM_PROMPT
    if FORMAT_DESCRIPTION:
        system_content = f"{system_content}\n\n{FORMAT_DESCRIPTION}" if system_content else FORMAT_DESCRIPTION

    if system_content:
        messages.append({"role": "system", "content": system_content})

    user_content = USER_PROMPT
    if PROMPT_MODIFIER:
        user_content += PROMPT_MODIFIER
    if COMPLETION_INSTRUCTION:
        user_content += f"\n\n{COMPLETION_INSTRUCTION}"

    messages.append({"role": "user", "content": user_content})

    return messages


def get_extra_params():
    """Возвращает дополнительные параметры для API."""
    return {"top_k": TOP_K} if TOP_K is not None else None


def print_api_params(messages, stop_list):
    """Выводит параметры API-запроса."""
    print("API CALL ARGS (sanitized):")
    print(f"  Model: {MODEL}")
    print(f"  Max tokens: {MAX_TOKENS}")
    print(f"  Temperature: {TEMPERATURE}")
    print(f"  Top-p: {TOP_P}")
    print(f"  Top-k: {TOP_K}")
    print(f"  Stop sequences: {stop_list}")
    print(f"  Messages count: {len(messages)}")
    print("  Messages preview:")
    for i, msg in enumerate(messages, 1):
        print(f"    Message {i}: role='{msg.get('role')}', content length={len(msg.get('content', ''))}")


def main():
    if not API_KEY:
        raise SystemExit("API_KEY environment variable is not set.")

    messages = build_messages()
    stop_list = [s.strip() for s in STOP_SEQUENCES] if STOP_SEQUENCES else None
    extra_params = get_extra_params()

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        stop=stop_list,
        extra_body=extra_params
    )

    print_api_params(messages, stop_list)
    print()
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
