# Use Case: Фоновый вывод уведомлений (Notification Watcher)

## Что нового

До этого изменения уведомления от планировщика появлялись **только перед следующим вводом** —
пользователю нужно было нажать Enter, чтобы увидеть напоминание.

После изменения уведомление появляется **немедленно**, даже если пользователь ничего не вводит.
Фоновый daemon-поток каждые 0.5 с проверяет очередь и печатает прямо в терминал.

---

## Способ 1: Ручное тестирование через чатбот + планировщик

### Шаг 1. Запустить чатбот

```bash
cd /Users/igorpotema/mycode/ai_ad_ch6/llm_agent
source ../.venv/bin/activate
python script.py
```

При старте вы увидите строку вида:
```
[Webhook: http://localhost:8088/notify]
```

Запомните порт (по умолчанию `8088`).

### Шаг 2. Создать напоминание через LLM

В prompt чатбота введите:
```
создай напоминание через 10 секунд: выпить воду
```

LLM должна вызвать MCP-инструмент планировщика, который зарегистрирует задачу
с callback на `http://localhost:8088/notify`.

### Шаг 3. Ничего не вводить

Просто ждите. Не нажимайте Enter.

### Ожидаемый результат (через ~10 секунд)

```
>
⏰ [REMINDER] выпить воду (id=<uuid>)
>
```

Курсор `> ` переиздаётся автоматически — накопленный текст ввода не теряется.

---

## Способ 2: Ручная отправка webhook без чатбота

Позволяет проверить механизм изолированно, без LLM.

### Терминал 1 — запустить чатбот

```bash
cd /Users/igorpotema/mycode/ai_ad_ch6/llm_agent
source ../.venv/bin/activate
python script.py
```

Убедитесь, что видите `[Webhook: http://localhost:8088/notify]`.

### Терминал 2 — отправить уведомление вручную

```bash
curl -s -X POST http://localhost:8088/notify \
  -H "Content-Type: application/json" \
  -d '{"task_id": "test-001", "description": "тест фонового уведомления"}'
```

Ожидаемый ответ curl:
```json
{"status":"ok"}
```

### Ожидаемый результат в Терминале 1 (мгновенно)

```
>
⏰ [REMINDER] тест фонового уведомления (id=test-001)
>
```

---

## Способ 3: Автотест (pytest)

```bash
cd /Users/igorpotema/mycode/ai_ad_ch6/llm_agent
source ../.venv/bin/activate
python -m pytest tests/test_notification_server.py::test_watcher_prints_notification_immediately -v
```

Ожидаемый вывод:
```
tests/test_notification_server.py::test_watcher_prints_notification_immediately PASSED
```

### Запустить все тесты notification_server

```bash
python -m pytest tests/test_notification_server.py -v
```

Ожидаемый вывод — 9 тестов, все `PASSED`.

---

## Способ 4: Минимальный Python-скрипт (без чатбота)

Позволяет проверить `_start_notification_watcher` в изоляции.

```python
# Запустить: python check_watcher.py
import time
import sys
sys.path.insert(0, "/Users/igorpotema/mycode/ai_ad_ch6/llm_agent")

from llm_agent.chatbot.notification_server import NotificationServer
from llm_agent.chatbot.main import _start_notification_watcher

srv = NotificationServer(port=9099)
srv.start()
time.sleep(0.05)

_start_notification_watcher(srv)
print(f"Сервер запущен: {srv.get_url()}")
print("Отправляем уведомление через 2 секунды...")
time.sleep(2)

import urllib.request, json
body = json.dumps({"task_id": "x42", "description": "проверка вотчера"}).encode()
req = urllib.request.Request(srv.get_url(), data=body, method="POST",
                              headers={"Content-Type": "application/json"})
urllib.request.urlopen(req)

print("Уведомление отправлено. Ждём печати (до 1 сек)...")
time.sleep(1)
```

Ожидаемый вывод:
```
Сервер запущен: http://localhost:9099/notify
Отправляем уведомление через 2 секунды...
Уведомление отправлено. Ждём печати (до 1 сек)...

⏰ [REMINDER] проверка вотчера (id=x42)
>
```

---

## Граничные случаи для ручной проверки

| Сценарий | Как воспроизвести | Ожидаемое поведение |
|----------|-------------------|---------------------|
| Несколько уведомлений подряд | Послать 3 curl-запроса с разницей < 0.5 с | Все 3 выводятся последовательно на следующем цикле вотчера |
| Уведомление пока пользователь набирает текст | Отправить curl, не нажимая Enter | Уведомление вставляется в терминал, накопленный ввод не теряется; Enter отправит его корректно |
| Порт занят (вотчер не стартует) | Занять порт 8088 перед запуском скрипта | Чатбот стартует без `[Webhook: ...]`, уведомлений нет, всё остальное работает |

---

## Архитектура (кратко)

```
scheduler/MCP ──POST /notify──► NotificationServer (HTTP :8088)
                                       │  queue.Queue (thread-safe)
                                       ▼
                           _notification_watcher() daemon thread
                                (poll каждые 0.5 с)
                                       │
                                       ▼
                           sys.stdout.write("\r\n⏰ note\n> ")
```

Ключевые файлы:
- `chatbot/notification_server.py` — HTTP-сервер + очередь (не изменялся)
- `chatbot/main.py` — `_start_notification_watcher()` + вызов после `start()`
- `tests/test_notification_server.py` — `test_watcher_prints_notification_immediately`
