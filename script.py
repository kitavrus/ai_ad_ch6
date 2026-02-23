from openai import OpenAI
import os
import time

# === НАСТРОЙКИ ===
# Базовый URL API
BASE_URL = "https://routerai.ru/api/v1"
# API ключ для авторизации
API_KEY = os.getenv("API_KEY")

# === Параметры модели ===
# Модель для генерации (выбранная модель)
#MODEL = "deepseek/deepseek-v3.2"
#MODEL = "qwen/qwen3.5-plus-02-15"
#MODEL = "anthropic/claude-sonnet-4.6"
MODEL = "gryphe/mythomax-l2-13b"
# Максимальное количество токенов в ответе
MAX_TOKENS = None
# Top-p: вероятность влияния на выбор следующего токена (0-1)
TOP_P = 0.9
# Top-k: ограничение по количеству кандидатов
TOP_K = 50
# Температура генерации (уровень случайности, 0-2)
TEMPERATURE = 0.7
# === Параметры промптов ===
# Системный промпт, задающий контекст ассистента //Ты полезный ассистент.
SYSTEM_PROMPT = ""
# Основной пользовательский промпт (начало беседы)
USER_PROMPT = "Предскажи курс рубля к тенге на 2026г"
# Модификатор к промпту, добавляющий инструкции к сообщению пользователя
PROMPT_MODIFIER = "Ответь максимально подробно с примерами"
# Описание формата ответа, добавляемое к промпту (например, JSON с полями)
FORMAT_DESCRIPTION = "" # Ответ должен быть в dm формате
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


def _get_usage_value(usage, key):
    """Safely extract a token usage value from a usage object/dict."""
    if usage is None:
        return None
    # support both dict-like and attribute-style access
    if isinstance(usage, dict):
        return usage.get(key)
    return getattr(usage, key, None)


def _format_metadata_line(model, endpoint, temp_val, ttft, req_time, total_s, total_tokens, p_tokens, c_tokens, cost_rub):
    """Return a single metadata line appended to the assistant's answer.
    ttft: time to first token in seconds
    req_time: duration of the actual API call execution in seconds
    total_s: total elapsed time for the whole operation in seconds
    """
    return (
        f"Model={model} | Endpoint={endpoint} | Temp=locked({temp_val:.1f}) | "
        f"TTFT={ttft:.3f}s | ReqTime={req_time:.3f}s | Total={total_s:.3f}s | Tokens={total_tokens} "
        f"(p={p_tokens}, c={c_tokens}) | Cost={cost_rub:.4f} ₽"
    )


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

    # Simple non-streaming path: get full response and print content only, plus metadata
    start_total = time.time()
    start_call = time.perf_counter()
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        stop=stop_list,
        extra_body=extra_params
    )
    api_call_time = time.perf_counter() - start_call
    total_s = time.time() - start_total

    # Print only the assistant content
    print(response.choices[0].message.content)

    # Compute usage metrics if available
    usage = getattr(response, "usage", None)
    prompt_tokens = int(_get_usage_value(usage, "prompt_tokens") or 0)
    completion_tokens = int(_get_usage_value(usage, "completion_tokens") or 0)
    total_tokens = int(_get_usage_value(usage, "total_tokens") or 0)
    if total_tokens == 0:
        total_tokens = prompt_tokens + completion_tokens

    # Simple cost estimation
    USD_PER_1K_TOKENS = 0.0015
    RUB_PER_USD = 100.0
    usd_cost = (total_tokens / 1000.0) * USD_PER_1K_TOKENS
    cost_rub = usd_cost * RUB_PER_USD

    # Print metadata line
    model_line = _format_metadata_line(
        MODEL,
        "chat",
        TEMPERATURE,
        api_call_time,
        api_call_time,
        total_s,
        total_tokens,
        prompt_tokens,
        completion_tokens,
        cost_rub,
    )
    print(model_line)
    return


if __name__ == "__main__":
    main()
