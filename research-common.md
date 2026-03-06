# Research: Предложения по оптимизации команд

## Выявленные проблемы

### Проблема 1: Перегрузка пространства имён `/plan`

**Текущее состояние**: `/plan` управляет тремя несвязанными вещами одновременно:
- Управление агентским режимом: `on`, `off`, `status`, `retries`
- Выполнение задач: `builder`
- Просмотр результатов: `result`

Это смешивает режим работы системы (agent mode) с операциями над объектами (task execution).

**Предложение**: Разделить на два namespace:
- `/agent on|off|status|retries` → управление режимом Stateful AI Agent
- `/plan builder|result|cancel` → выполнение и просмотр плана
- Добавить `/plan cancel` для отмены FSM-диалога из любого состояния

**Сложность**: Высокая (ломает обратную совместимость). **Приоритет**: Низкий.

---

### Проблема 2: Несогласованность команд памяти

**Текущее состояние**: Три команды с разными стилями именования:
- `/settask TEXT` — рабочая память, не входит в `/mem*` семейство
- `/setpref KEY=VALUE` — рабочая память, аналогично
- `/remember KEY: VALUE` — долгосрочная память, аналогично

**Предложение**: Добавить алиасы под единым namespace `/mem`:
- `/mem task TEXT` → alias для `/settask`
- `/mem pref KEY=VALUE` → alias для `/setpref`
- `/mem know KEY: VALUE` → alias для `/remember`
- Старые команды сохраняются (обратная совместимость)

**Сложность**: Средняя. **Приоритет**: Средний.

---

### Проблема 3: Ручное vs автоматическое выполнение задач

**Текущее состояние**: Два несовместимых способа выполнить план:
- Ручной: `/task start` + `/task step done TEXT`
- Автоматический: `/plan builder`

Команды находятся в разных namespace (`/task` и `/plan`), хотя оперируют одним объектом.

**Предложение**: Добавить `/task execute` как alias для `/plan builder` (с опцией `--plan NAME`):
```
/task execute           → alias /plan builder (для активного плана)
/task execute --plan X  → alias /plan builder --plan X
```
Документировать два режима: "ручной" (`/task step`) и "автоматический LLM" (`/task execute`).

**Сложность**: Низкая. **Приоритет**: Высокий.

---

### Проблема 4: Слабая интеграция `/project` с задачами

**Текущее состояние**:
- `/project add-plan TASK_ID` — требует UUID (не имя); неудобно в работе
- Нет способа просмотреть задачи внутри проекта
- Нет способа создать задачу сразу с привязкой к проекту
- Нет способа переименовать задачу или изменить её описание

**Предложение**: Добавить два симметричных подпространства `/project task` и `/project plan`:

| Команда | Действие |
|---------|---------|
| `/project tasks` | Показать все задачи/планы проекта |
| `/project task new DESC` | Создать задачу и автоматически привязать к активному проекту |
| `/project task rename NEW_NAME [--plan NAME]` | Переименовать задачу |
| `/project task describe TEXT [--plan NAME]` | Обновить описание задачи |
| `/project plans` | Alias → `/project tasks` |
| `/project plan new DESC` | Alias → `/project task new` |
| `/project plan rename NEW_NAME [--plan NAME]` | Alias → `/project task rename` |
| `/project plan describe TEXT [--plan NAME]` | Alias → `/project task describe` |
| `/project add-plan TASK_ID` | Привязать существующую задачу по UUID (существующая команда) |
| `/project add-plan-name NAME` | Привязать существующую задачу по имени (новая) |

**Ключевое дизайн-решение**: `/project task *` и `/project plan *` — взаимозаменяемые синонимы,
т.к. в кодовой базе `TaskPlan` = план задачи. Это сохраняет согласованность с уже существующим
разделением namespace `/task` (глобальные операции) и `/plan` (agentные операции).

**Сложность**: Средняя. **Приоритет**: Высокий.

---

### Проблема 5: Видимость состояния контекста

**Текущее состояние**:
- `/showsummary` работает только для `sliding_window` (выводит summary)
- Для других стратегий нет аналога
- Нет способа увидеть текущий размер контекста / количество сообщений

**Предложение**:
- `/strategy status` — показать текущую стратегию + параметры + размер контекста:
  ```
  Стратегия: sliding_window
  Сообщений в памяти: 42
  В окне (recent_n): 10
  Summary: [есть / нет]
  ```
- Для `sticky_facts`: показать количество фактов
- Для `branching`: показать активную ветку и размер

**Сложность**: Низкая. **Приоритет**: Средний.

---

### Проблема 6: FSM Plan-диалог без возможности отмены

**Текущее состояние**: После `/plan on` система входит в FSM. Выйти из состояний
`awaiting_task`, `awaiting_invariants`, `confirming` можно только через `/plan off`
(который требует быть в "обычном" режиме обработки команд, что работает).

Однако отсутствует явная команда отмены, что неочевидно для пользователя.

**Предложение**:
- `/plan cancel` → сбросить `plan_dialog_state = None`, очистить `plan_draft_steps = []`
- При входе в каждое FSM-состояние показывать: "Введите /plan cancel для отмены"

**Сложность**: Низкая (1 ветка в `_handle_agent_command`). **Приоритет**: Высокий.

---

## Приоритизация

| Приоритет | Предложение | Сложность | Файлы |
|-----------|------------|-----------|-------|
| Высокий | `/plan cancel` — отмена FSM | Низкая | cli.py, main.py |
| Высокий | `/project tasks`, `/project task new|rename|describe` | Средняя | cli.py, main.py |
| Высокий | `/project plans`, `/project plan new|rename|describe` (aliases) | Низкая | cli.py |
| Высокий | `/project add-plan-name NAME` | Низкая | cli.py, main.py |
| Высокий | `/task execute` alias для `/plan builder` | Низкая | cli.py |
| Средний | `/strategy status` | Низкая | cli.py, main.py |
| Средний | `/mem task|pref|know` aliases | Средняя | cli.py |
| Низкий | Разделение `/plan` → `/agent` + `/plan` | Высокая | cli.py, main.py, docs |

---

## Итоговая карта команд (целевое состояние)

```
ПАРАМЕТРЫ МОДЕЛИ         /model /base-url /temperature /max-tokens /top-p /top-k
                         /system-prompt /initial-prompt

КОНТЕКСТ                 /strategy [sw|sf|br]
                         /strategy status                          [НОВОЕ]
                         /showsummary /showfacts
                         /setfact /delfact
                         /checkpoint /branch /switch /branches

ПАМЯТЬ                   /memshow /memstats /memclear /memsave /memload
                         /settask /setpref /remember               [существующие]
                         /mem task /mem pref /mem know             [НОВЫЕ aliases]

ПРОФИЛЬ                  /profile show|list|name|style|format|constraint|load

ЗАДАЧИ (глобальные)      /task new|show|list|load|delete
                         /task start|step|pause|resume|done|fail|result
                         /task execute                             [НОВОЕ alias]

ПЛАН (agentные)          /plan on|off|status|retries|builder|result
                         /plan cancel                              [НОВОЕ]

ИНВАРИАНТЫ               /invariant add|del|edit|list|clear

ПРОЕКТЫ (управление)     /project new|list|show|switch|delete
ПРОЕКТЫ (задачи)         /project tasks                           [НОВОЕ]
                         /project task new|rename|describe         [НОВЫЕ]
                         /project add-plan [TASK_ID]              [существующая]
                         /project add-plan-name [NAME]            [НОВОЕ]
ПРОЕКТЫ (планы, aliases) /project plans                           [НОВЫЙ alias]
                         /project plan new|rename|describe         [НОВЫЕ aliases]

СЕССИЯ                   /resume
```
