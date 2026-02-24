-Project Title
ИИ CLI чат с сохранением диалога

Overview
Эта программа реализует интерфейс командной строки (CLI) для общения с API, совместимым с OpenAI. Она сохраняет историю диалога, переписки на диск и пишет per‑request метрики в отдельные логи. Поддерживается возобновление сессии и продолжение диалога с учётом контекста.

Цели
- Предоставить надёжный интерактивный CLI для чата с моделью ИИ.
- Сохранять одну сессию как единый JSON файл и постепенно обновлять его во время диалога.
- Создавать per‑request логи метрик в отдельной директории, по одному файлу на запрос.
- Предусмотреть возможность возобновления последней сессии и продолжения диалога.
- Обеспечить чистую и хорошо документированную кодовую базу для воспроизведения другим ИИ.

Архитектура
- CLI Frontend (script.py)
  - Разбор аргументов командной строки: модель, URL API, параметры генерации, промпты и флаг возобновления.
  - Формирует начальные системный и пользовательский промпты, затем входит в интерактивный цикл.
  - Принимает ввод пользователя, добавляет его к диалогу, отправляет запрос к API и выводит ответ ассистента, обновляя сессию.
  - При выходе сохраняет финальную сессию и выводит сводку.
- API Client Interaction
  - Использует OpenAI‑совместимый клиент для отправки запросов chat completions.
  - Поддерживает дополнительные поля, например top_k через аргумент extra_body.
- Persistence Layer
  - Диалоги: dialogues/dialogue_<timestamp>_<model>.json
  - Сессия: dialogues/session_<timestamp>_<model>.json (один файл на запуск; перезаписывается)
  - Метрики per-request: dialogues/metrics/session_<session_id>_req_<idx>.log (JSON)
- Метаданные и журналирование
  - Метрики: TTFT, ReqTime, TotalTime, Tokens, p_tokens, c_tokens, Cost
  - per-request логи — JSON с перечисленными полями
- Поддержка возобновления
  - CLI‑флаг --resume позволяет загрузить последнюю сохранённую сессию и продолжить диалог
- Расширяемость
  - Легко разделить на модули (api_client, persistence, cli) и добавить тесты
- Данные на диске
  - Dialogue file: dialogues/dialogue_<ts>_<model>.json
  - Session file: dialogues/session_<ts>_<model>.json
  - Per-request log: dialogues/metrics/session_<session_id>_req_<idx>.log

CLI интерфейс
- Аргументы: --model/-m, --base-url/-u, --max-tokens, --temperature/-T, --top-p/-p, --top-k/-k, --system-prompt, --initial-prompt, --resume
- Возможности: возобновление сессии, интерактивный диалог, логирование метрик

Использование
- Начать новую сессию: python script.py
- Возобновить последнюю сессию: python script.py --resume
- Настроить модель и параметры: python script.py -m "ваша_модель" -T 0.8
- Логи метрик сохраняются в dialogues/metrics/

Что делает код
- Инициализация клиента OpenAI с API‑ключом и базовым URL
- Формирование начальных сообщений (system_prompt, initial_prompt)
- Вход в интерактивный цикл: ввод пользователя, вызов API, печать ответа, обновление переписки
- Сохранение per-request метрик в dialogues/metrics/
- Обновление файла сессии dialogues/session_*.json на каждый шаг
- По выходу сохраняется финальная версия сессии

Как использовать это описание для генерации кода другими ИИ
- Этот файл описывает требования и формат файлов; можно использовать как единый паттерн для генерации кода
- При генерации кода можно расширять и создавать модули: api_client.py, persistence.py, cli.py

End of file
- Dialogue file (dialogues/dialogue_<ts>_<model>.json)
  - dialogue_session_id
  - created_at
  - model
  - base_url
  - system_prompt
  - initial_prompt
  - messages: array of {role, content}
  - turns: number of user/assistant turns
  - last_user_input
  - last_assistant_content
  - duration_seconds
  - requests: array of per-request metadata (optional or incrementally built)
- Session file (dialogues/session_<ts>_<model>.json)
  - session-level metadata (same as above)
  - messages
  - turns
  - duration_seconds
- Per-request metric log (dialogues/metrics/session_<session_id>_req_<idx>.log)
  - JSON object with fields:
    - model
    - endpoint
    - temp
    - ttft
    - req_time
    - total_time
    - tokens
    - p_tokens
    - c_tokens
    - cost_rub

CLI Interface (arguments and defaults)
- --model / -m: model to use (default: DEFAULT_MODEL)
- --base-url / -u: base API URL (default: BASE_URL)
- --max-tokens: max tokens in response (default: None)
- --temperature / -T: generation temperature (default: DEFAULT_TEMPERATURE)
- --top-p / -p: nucleus sampling (default: DEFAULT_TOP_P)
- --top-k / -k: top-k (default: DEFAULT_TOP_K)
- --system-prompt: system prompt text
- --initial-prompt: initial user prompt (seed)
- --resume: load last session and resume dialog (default: False)

Usage Scenarios
- Start a new session:
  - python script.py
- Resume last session and continue:
  - python script.py --resume
- Customize model and generation:
  - python script.py -m "qwen/qwen3.5-plus-02-15" -T 0.8
- Save and log per-request metrics:
  - Automatically enabled by the code; per-request logs go to dialogues/metrics/

What the code does (high-level)
- Initialize OpenAI client with API key and base URL.
- Build initial messages with system_prompt and initial_prompt if provided.
- Enter interactive loop:
  - Read user input
  - Append to messages with role "user"
  - Send to API as a chat completion request
  - Print the assistant's reply
  - Append reply to messages with role "assistant"
  - Save a per-step metric to dialogues/metrics/
  - Update a single session file dialogues/session_*.json with the current dialogue state
- On exit, save final session state and log the path.

How to use this spec with another AI
- This file is self-contained; it can be used by another AI to reproduce the same code structure.

End of file
