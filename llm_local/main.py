import ollama

DEFAULT_MODEL = "qwen3:14b"
DEFAULT_TEMPERATURE = 0.7

client = ollama.Client()


def ask_llm(prompt: str, model: str, temperature: float):
    resp = client.generate(
        model=model,
        prompt=prompt,
        options={"temperature": temperature}
    )
    return resp["response"]


if __name__ == "__main__":
    current_model = DEFAULT_MODEL
    current_temperature = DEFAULT_TEMPERATURE

    print(f"Модель: {current_model} | температура: {current_temperature}")
    print("Команды: /model <name>, /temperature <0-2>, /status, /help, exit")

    while True:
        text = input("\n> ").strip()
        if not text:
            continue
        if text.lower() == "exit":
            break
        elif text.startswith("/model "):
            current_model = text[7:].strip()
            print(f"Модель: {current_model}")
        elif text.startswith("/temperature "):
            val = text[13:].strip()
            try:
                t = float(val)
                if not 0.0 <= t <= 2.0:
                    print("Ошибка: температура должна быть от 0.0 до 2.0")
                else:
                    current_temperature = t
                    print(f"Температура: {current_temperature}")
            except ValueError:
                print(f"Ошибка: '{val}' не число")
        elif text == "/status":
            print(f"Модель: {current_model} | температура: {current_temperature}")
        elif text == "/help":
            print("/model <name>        — сменить модель")
            print("/temperature <0-2>   — сменить температуру")
            print("/status              — текущие настройки")
            print("exit                 — выйти")
        else:
            answer = ask_llm(text, current_model, current_temperature)
            print(f"\n{current_model}:", answer)
