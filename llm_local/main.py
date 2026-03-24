import ollama

client = ollama.Client()

def ask_qwen3(prompt: str):
    resp = client.generate(
        model="qwen3:14b",
        prompt=prompt,
        options={"temperature": 0.7}
    )
    return resp["response"]

if __name__ == "__main__":
    print("Qwen3-14b готов. Введите вопрос (или 'exit' для выхода).")
    while True:
        text = input("\n> ").strip()
        if not text or text.lower() == "exit":
            break
        answer = ask_qwen3(text)
        print("\nQwen3-14b:", answer)
