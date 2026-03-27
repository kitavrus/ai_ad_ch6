import sys
import time
import threading
import ollama

DEFAULT_MODEL = "qwen3:14b"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 2048
DEFAULT_CTX = 4096

PRESETS = {
    "architect": (
        "You are a software architect. When asked to design or describe a system:\n"
        "- Use C4 Model notation (Context → Container → Component → Code levels)\n"
        "- Produce Data Flow Diagrams (DFD) in structured text using → arrows and trust boundaries\n"
        "- Produce Sequence Diagrams in PlantUML or Mermaid syntax\n"
        "- Always clarify: system context, actors, external systems, data flows, trust zones\n"
        "- Output diagrams as fenced code blocks (```plantuml or ```mermaid)\n"
        "- Be concise: one diagram per response unless asked for all levels\n"
        "- IMPORTANT: Always respond in Russian language."
    ),
    "analyst": (
        "You are a business analyst. When asked about a system or feature:\n"
        "- Write user stories in format: As a <role>, I want <goal>, so that <benefit>\n"
        "- Define acceptance criteria as a numbered checklist (Given/When/Then or simple list)\n"
        "- Describe business processes as BPMN-style text flows with actors, gateways, and outcomes\n"
        "- Identify stakeholders, their goals, and pain points\n"
        "- Highlight risks, assumptions, and open questions\n"
        "- IMPORTANT: Always respond in Russian language."
    ),
    "tester": (
        "You are a QA engineer. When asked about a system or feature:\n"
        "- Write test plans with scope, objectives, and test types (functional, regression, edge)\n"
        "- Produce test cases in format: ID | Описание | Шаги | Ожидаемый результат | Приоритет\n"
        "- Always include edge cases, boundary values, and negative scenarios\n"
        "- Suggest test automation approaches (unit, integration, e2e) with tool recommendations\n"
        "- Identify what cannot be tested and why\n"
        "- IMPORTANT: Always respond in Russian language."
    ),
    "devops": (
        "You are a DevOps engineer. When asked about infrastructure or deployment:\n"
        "- Design CI/CD pipelines with concrete stages (build, test, scan, deploy, rollback)\n"
        "- Provide Docker/Kubernetes configs as fenced code blocks (```yaml or ```dockerfile)\n"
        "- Include monitoring, alerting, and observability (metrics, logs, traces)\n"
        "- Consider security: secrets management, network policies, least privilege\n"
        "- Always mention rollback and disaster recovery strategy\n"
        "- IMPORTANT: Always respond in Russian language."
    ),
    "pm": (
        "You are a project manager. When asked about a project or feature:\n"
        "- Build a roadmap with phases, milestones, and deliverables\n"
        "- Identify risks with probability/impact matrix and mitigation strategies\n"
        "- Define success metrics (KPIs) and how to measure them\n"
        "- Describe team communication plan: ceremonies, artifacts, escalation paths\n"
        "- Estimate effort in story points or t-shirt sizes, not calendar time\n"
        "- IMPORTANT: Always respond in Russian language."
    ),
    "developer": (
        "You are a senior software developer. When asked about code or implementation:\n"
        "- Write clean, idiomatic code following SOLID principles\n"
        "- Apply appropriate design patterns and explain the choice\n"
        "- Provide code review feedback: what to improve and why\n"
        "- Suggest refactoring steps with before/after examples\n"
        "- Always include error handling, input validation, and basic tests\n"
        "- IMPORTANT: Always respond in Russian language."
    ),
    "default": "",
}

BENCHMARK_QUESTIONS = [
    ("architecture", "Спроектируй простой сервис обработки платежей. Покажи C4 Context диаграмму."),
    ("coding",       "Напиши функцию на Python для разбора JWT-токена без сторонних библиотек."),
    ("general",      "Объясни теорему CAP в трёх предложениях."),
]

client = ollama.Client()


def ask_llm(
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    ctx: int,
    system_prompt: str = "",
) -> tuple[str, bool]:
    """Returns (answer, truncated). truncated=True if response was cut by token limit."""
    options = {
        "temperature": temperature,
        "num_predict": max_tokens,
        "num_ctx": ctx,
    }
    if system_prompt:
        resp = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": prompt},
            ],
            options=options,
        )
        truncated = resp.get("done_reason") == "length"
        return resp["message"]["content"], truncated
    else:
        resp = client.generate(model=model, prompt=prompt, options=options)
        truncated = resp.get("done_reason") == "length"
        return resp["response"], truncated


def _parse_settings(text: str) -> tuple[dict, list[str]]:
    """Parse 'key=value ...' pairs. Returns (updates, errors)."""
    updates: dict = {}
    errors: list[str] = []
    for token in text.split():
        if "=" not in token:
            errors.append(f"Игнорирую '{token}' (нет '=')")
            continue
        key, _, val = token.partition("=")
        key = key.strip().lower()
        val = val.strip()
        if key == "temperature":
            try:
                v = float(val)
                if not 0.0 <= v <= 2.0:
                    errors.append("temperature должна быть 0.0–2.0")
                else:
                    updates["temperature"] = v
            except ValueError:
                errors.append(f"temperature: '{val}' не число")
        elif key == "max_tokens":
            try:
                v = int(val)
                if v < 1:
                    errors.append("max_tokens должно быть ≥ 1")
                else:
                    updates["max_tokens"] = v
            except ValueError:
                errors.append(f"max_tokens: '{val}' не целое число")
        elif key == "ctx":
            try:
                v = int(val)
                if v < 512:
                    errors.append("ctx должно быть ≥ 512")
                else:
                    updates["ctx"] = v
            except ValueError:
                errors.append(f"ctx: '{val}' не целое число")
        else:
            errors.append(f"Неизвестный ключ '{key}' (доступны: temperature, max_tokens, ctx)")
    return updates, errors


def _show_models() -> None:
    try:
        models = client.list()["models"]
    except Exception as e:
        print(f"Ошибка получения списка моделей: {e}")
        return
    if not models:
        print("Нет загруженных моделей. Используйте: ollama pull <model>")
        return
    print(f"{'Модель':<40} {'Размер':>10}")
    print("-" * 52)
    for m in models:
        name = m.get("name", "?")
        size = m.get("size", 0)
        size_gb = f"{size / 1e9:.1f} GB" if size else "?"
        print(f"{name:<40} {size_gb:>10}")
    print("\nРекомендации по квантованию (qwen3:14b):")
    print("  qwen3:14b-q4_K_M  — Q4, ~8GB, скорость +40%, качество ~95%")
    print("  qwen3:14b-q8_0    — Q8, ~15GB, качество ≈ fp16")
    print("  qwen3:8b-q4_K_M   — если RAM < 16GB")
    print("Установить: ollama pull <model-name>")


def _spinner(stop_event: threading.Event, t0: float, label: str) -> None:
    frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    i = 0
    while not stop_event.is_set():
        elapsed = time.perf_counter() - t0
        sys.stdout.write(f"\r  {frames[i % len(frames)]} [{label}] думает... {elapsed:.1f}s")
        sys.stdout.flush()
        i += 1
        time.sleep(0.1)
    sys.stdout.write("\r" + " " * 50 + "\r")
    sys.stdout.flush()


def _run_benchmark(model: str, temperature: float, max_tokens: int, ctx: int, system_prompt: str, preset_name: str = "default") -> None:
    import os
    from datetime import datetime

    preset_label = preset_name
    started_at = datetime.now()
    ts = started_at.strftime("%Y%m%d_%H%M%S")
    safe_model = model.replace(":", "_").replace("/", "_")

    os.makedirs("benchmark_results", exist_ok=True)
    path = f"benchmark_results/{ts}_{safe_model}_{preset_label}.md"

    def w(text: str) -> None:
        """Дописать строку в файл и сбросить буфер."""
        with open(path, "a", encoding="utf-8") as f:
            f.write(text)

    header = (
        f"# Benchmark Report\n\n"
        f"| Параметр | Значение |\n"
        f"|----------|----------|\n"
        f"| Модель | `{model}` |\n"
        f"| temperature | `{temperature}` |\n"
        f"| max_tokens | `{max_tokens}` |\n"
        f"| ctx | `{ctx}` |\n"
        f"| Preset | `{preset_label}` |\n"
        f"| Дата | {started_at.strftime('%Y-%m-%d %H:%M:%S')} |\n\n"
        f"---\n\n"
    )
    print(f"\n{header}")
    w(header)

    total_chars = 0
    total_time = 0.0

    for i, (label, question) in enumerate(BENCHMARK_QUESTIONS, 1):
        section_header = f"## {i}. [{label}]\n\n**Вопрос:** {question}\n\n"
        print(f"[{label}] {question}")
        w(section_header)

        t0 = time.perf_counter()
        stop_event = threading.Event()
        spinner_thread = threading.Thread(target=_spinner, args=(stop_event, t0, label), daemon=True)
        spinner_thread.start()
        try:
            answer, truncated = ask_llm(question, model, temperature, max_tokens, ctx, system_prompt)
        except Exception as e:
            stop_event.set()
            spinner_thread.join()
            err_line = f"> **Ошибка:** {e}\n\n---\n\n"
            print(f"  Ошибка: {e}\n")
            w(err_line)
            continue
        finally:
            stop_event.set()
            spinner_thread.join()

        elapsed = time.perf_counter() - t0
        chars = len(answer)
        total_chars += chars
        total_time += elapsed

        trunc_warn = ""
        if truncated:
            trunc_warn = f"  ⚠️  ОТВЕТ ОБРЕЗАН — закончились токены (max_tokens={max_tokens}). Увеличьте: /settings max_tokens=4096"
            print(trunc_warn)

        stats = f"  Время: {elapsed:.1f}s | Символов: {chars} | ~{chars/elapsed:.0f} сим/с"
        print(stats)
        print(f"  Ответ:\n{answer}\n")

        trunc_md = f"\n> ⚠️ **Ответ обрезан** — закончились токены (`max_tokens={max_tokens}`). Увеличьте: `/settings max_tokens=4096`\n" if truncated else ""
        w(
            f"**Статистика:** время `{elapsed:.1f}s` | символов `{chars}` | скорость `{chars/elapsed:.0f} сим/с`\n"
            f"{trunc_md}\n"
            f"**Ответ:**\n\n{answer}\n\n---\n\n"
        )

    if total_time > 0:
        summary_line = f"Итого: {total_time:.1f}s | Средняя скорость: {total_chars/total_time:.0f} сим/с"
        print(summary_line)
        w(f"## Итого\n\n`{summary_line}`\n")

    print(f"\nРезультаты сохранены: {path}")


def _run_benchmark_all(model: str, temperature: float, max_tokens: int, ctx: int) -> None:
    import os
    from datetime import datetime

    started_at = datetime.now()
    ts = started_at.strftime("%Y%m%d_%H%M%S")
    safe_model = model.replace(":", "_").replace("/", "_")
    os.makedirs("benchmark_results", exist_ok=True)
    path = f"benchmark_results/{ts}_compare_{safe_model}.md"

    def w(text: str) -> None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(text)

    presets_to_run = [(name, prompt) for name, prompt in PRESETS.items()]
    n_presets = len(presets_to_run)
    n_questions = len(BENCHMARK_QUESTIONS)

    w(
        f"# Сравнительный отчёт пресетов\n\n"
        f"| Параметр | Значение |\n"
        f"|----------|----------|\n"
        f"| Модель | `{model}` |\n"
        f"| temperature | `{temperature}` |\n"
        f"| max_tokens | `{max_tokens}` |\n"
        f"| ctx | `{ctx}` |\n"
        f"| Пресетов | {n_presets} |\n"
        f"| Дата | {started_at.strftime('%Y-%m-%d %H:%M:%S')} |\n\n"
        f"---\n\n"
    )
    print(f"\n=== Benchmark ALL: {n_presets} пресетов × {n_questions} вопросов ===\n")

    # stats[preset_name] = [(chars, elapsed), ...]
    stats: dict[str, list[tuple[int, float]]] = {}

    for qi, (qlabel, question) in enumerate(BENCHMARK_QUESTIONS, 1):
        w(f"## Вопрос {qi}: {question}\n\n")
        print(f"\n── Вопрос {qi}/{n_questions}: {question}\n")

        for pi, (pname, pprompt) in enumerate(presets_to_run, 1):
            spinner_label = f"{pi}/{n_presets} {pname}"
            print(f"  [{spinner_label}]", end="", flush=True)

            t0 = time.perf_counter()
            stop_event = threading.Event()
            spinner_thread = threading.Thread(
                target=_spinner, args=(stop_event, t0, spinner_label), daemon=True
            )
            spinner_thread.start()
            try:
                answer, truncated = ask_llm(question, model, temperature, max_tokens, ctx, pprompt)
            except Exception as e:
                stop_event.set()
                spinner_thread.join()
                print(f"\n  Ошибка [{pname}]: {e}")
                w(f"### {pname}\n\n> **Ошибка:** {e}\n\n---\n\n")
                stats.setdefault(pname, []).append((0, 0.0))
                continue
            finally:
                stop_event.set()
                spinner_thread.join()

            elapsed = time.perf_counter() - t0
            chars = len(answer)
            stats.setdefault(pname, []).append((chars, elapsed))

            trunc_md = (
                f"\n> ⚠️ **Ответ обрезан** (`max_tokens={max_tokens}`). `/settings max_tokens=4096`\n"
                if truncated else ""
            )
            print(f"\n  [{pname}] {elapsed:.1f}s | {chars} сим")
            w(
                f"### {pname}\n\n"
                f"**Время:** `{elapsed:.1f}s` | **Символов:** `{chars}` | **Скорость:** `{chars/elapsed:.0f} сим/с`\n"
                f"{trunc_md}\n"
                f"{answer}\n\n---\n\n"
            )

    # Итоговая сводная таблица
    header_cols = " | ".join(f"Вопрос {i+1} (сим)" for i in range(n_questions))
    w(f"## Итоговая таблица\n\n| Пресет | {header_cols} | Всего (s) |\n")
    w(f"|--------|{'|'.join(['---'] * n_questions)}|---|\n")

    print("\n── Итого:")
    for pname, results in stats.items():
        cols = " | ".join(str(c) for c, _ in results)
        total_s = sum(e for _, e in results)
        w(f"| {pname} | {cols} | {total_s:.1f}s |\n")
        print(f"  {pname:<12} {' | '.join(str(c) for c, _ in results)}  итого {total_s:.1f}s")

    w("\n")
    print(f"\nРезультаты сохранены: {path}")


if __name__ == "__main__":
    import os
    from datetime import datetime

    current_model       = DEFAULT_MODEL
    current_temperature = DEFAULT_TEMPERATURE
    current_max_tokens  = DEFAULT_MAX_TOKENS
    current_ctx         = DEFAULT_CTX
    current_preset      = "default"
    current_system      = ""

    # Файл сессии чата
    os.makedirs("chat_logs", exist_ok=True)
    _session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    _chat_log_path = f"chat_logs/{_session_ts}_chat.md"

    def _chat_log(role: str, content: str, note: str = "") -> None:
        with open(_chat_log_path, "a", encoding="utf-8") as f:
            ts = datetime.now().strftime("%H:%M:%S")
            prefix = f"**[{ts}] {role}**"
            if note:
                prefix += f" _{note}_"
            f.write(f"{prefix}\n\n{content}\n\n---\n\n")

    def _status() -> None:
        print(
            f"Модель: {current_model} | температура: {current_temperature} "
            f"| max_tokens: {current_max_tokens} | ctx: {current_ctx} "
            f"| preset: {current_preset}"
        )

    _status()
    print("Команды: /model, /settings, /preset, /models, /benchmark, /status, /help, exit")
    print(f"Лог сессии: {_chat_log_path}")

    # Записать заголовок сессии
    with open(_chat_log_path, "w", encoding="utf-8") as f:
        f.write(
            f"# Сессия чата\n\n"
            f"| Параметр | Значение |\n"
            f"|----------|----------|\n"
            f"| Модель | `{current_model}` |\n"
            f"| temperature | `{current_temperature}` |\n"
            f"| max_tokens | `{current_max_tokens}` |\n"
            f"| ctx | `{current_ctx}` |\n"
            f"| Дата | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |\n\n"
            f"---\n\n"
        )

    while True:
        text = input("\n> ").strip()
        if not text:
            continue

        if text.lower() == "exit":
            break

        elif text == "/model":
            print("Использование: /model <name>")
            print("Пример: /model qwen3:14b-q4_K_M")
            print("Совет: /models — посмотреть список доступных моделей")

        elif text.startswith("/model "):
            current_model = text[7:].strip()
            print(f"Модель: {current_model}")

        elif text == "/settings":
            print("Использование: /settings key=value [key=value ...]")
            print("Ключи:")
            print(f"  temperature  — случайность ответа (0.0–2.0)  сейчас: {current_temperature}")
            print(f"  max_tokens   — макс. длина ответа             сейчас: {current_max_tokens}")
            print(f"  ctx          — размер контекстного окна       сейчас: {current_ctx}")
            print("Пример: /settings temperature=0.3 max_tokens=4096 ctx=8192")

        elif text.startswith("/settings "):
            updates, errors = _parse_settings(text[10:].strip())
            for e in errors:
                print(f"  ! {e}")
            if "temperature" in updates:
                current_temperature = updates["temperature"]
            if "max_tokens" in updates:
                current_max_tokens = updates["max_tokens"]
            if "ctx" in updates:
                current_ctx = updates["ctx"]
            if updates:
                _status()

        elif text == "/preset":
            print("Использование: /preset <name>")
            print("Доступные пресеты:")
            for pname, pcontent in PRESETS.items():
                if pcontent:
                    first_line = pcontent.split("\n")[0]
                    print(f"  {pname:<12} — {first_line[:60]}")
                else:
                    print(f"  {pname:<12} — без системного промпта (по умолчанию)")

        elif text.startswith("/preset "):
            name = text[8:].strip().lower()
            if name in ("off", "none"):
                name = "default"
            if name not in PRESETS:
                print(f"Неизвестный пресет '{name}'. Доступны: {', '.join(PRESETS)}")
            else:
                current_preset = name
                current_system = PRESETS[name]
                print(f"Пресет: {current_preset}")
                if current_system:
                    print(f"Системный промпт активен ({len(current_system)} символов):")
                    print("-" * 40)
                    print(current_system)
                    print("-" * 40)

        elif text == "/status":
            _status()

        elif text == "/models":
            _show_models()

        elif text == "/benchmark":
            _run_benchmark(
                current_model, current_temperature,
                current_max_tokens, current_ctx, current_system,
                preset_name=current_preset,
            )

        elif text == "/benchmark-all":
            _run_benchmark_all(
                current_model, current_temperature,
                current_max_tokens, current_ctx,
            )

        # Обратная совместимость: старая команда /temperature
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

        elif text == "/help":
            print("/model <name>                    — сменить модель")
            print("/settings temperature=0.6 max_tokens=1024 ctx=8192")
            print("                                 — настроить параметры генерации")
            print("/preset architect|default        — системный промпт (C4/DFD/SD или пусто)")
            print("/models                          — список моделей + советы по квантованию")
            print("/benchmark                       — тест текущего пресета (3 вопроса)")
            print("/benchmark-all                   — сравнение всех пресетов на одних вопросах")
            print("/status                          — текущие настройки")
            print("exit                             — выйти")
            print()
            print("Квантование: ollama pull qwen3:14b-q4_K_M  → /model qwen3:14b-q4_K_M")

        else:
            if text.startswith("/"):
                print(f"Неизвестная команда: '{text}'")
                print("Введите /help для списка команд.")
                continue
            _chat_log("Пользователь", text)
            t0 = time.perf_counter()
            stop_event = threading.Event()
            spinner_thread = threading.Thread(
                target=_spinner, args=(stop_event, t0, current_model), daemon=True
            )
            spinner_thread.start()
            try:
                answer, truncated = ask_llm(
                    text, current_model, current_temperature,
                    current_max_tokens, current_ctx, current_system,
                )
            except Exception as e:
                stop_event.set()
                spinner_thread.join()
                print(f"Ошибка: {e}")
                _chat_log("Ошибка", str(e))
                continue
            finally:
                stop_event.set()
                spinner_thread.join()

            elapsed = time.perf_counter() - t0
            print(f"\n{current_model} ({elapsed:.1f}s):", answer)
            trunc_note = f"⚠️ ответ обрезан, max_tokens={current_max_tokens}" if truncated else ""
            if truncated:
                print(f"\n⚠️  ОТВЕТ ОБРЕЗАН — закончились токены (max_tokens={current_max_tokens}). Увеличьте: /settings max_tokens=4096")
            _chat_log(current_model, answer, note=trunc_note)
