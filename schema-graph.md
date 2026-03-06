# Schema Graph: Command Dependencies

## Уровень 0: Глобальное пространство (profile_name)

```
dialogues/{profile_name}/        <- ВСЁ хранится здесь
     │
     ├── profile.json            <- UserProfile
     ├── session_*.json          <- SessionState snapshot
     ├── metrics/                <- RequestMetric логи
     ├── memory/
     │     ├── short_term/
     │     ├── working/
     │     └── long_term/
     ├── tasks/{task_id}/
     │     ├── plan.json
     │     └── step_NNN.json
     └── projects/{project_id}/
           └── project.json
```

Команды управления профилем:

```
/profile load <name>  ----------- меняет profile_name -> сбрасывает:
                                    agent_mode, plan_dialog_state,
                                    active_task_id, branches, токены
/profile name|style|format|      - меняет UserProfile в LongTermMemory
         constraint|model
--profile NAME (CLI arg)  -------- задаёт profile_name на старте
                                    + автоматически resume последней сессии
```

---

## Уровень 1: Сессионные параметры (SessionState, session_*.json)

```
/model /temperature /max-tokens   - меняют SessionState напрямую
/top-p /top-k /system-prompt       -> влияют на КАЖДЫЙ API-запрос
/strategy <sw|sf|br>  ------------ меняет context_strategy
                                    -> определяет какой контекст-билдер
                                       используется в API-запросе
```

---

## Уровень 2: Контекстные стратегии (взаимоисключающие)

```
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  sliding_window  │  │  sticky_facts    │  │   branching      │
│  (default)       │  │                  │  │                  │
│                  │  │ /showfacts        │  │ /checkpoint -----┼--┐
│  context_summary │  │ /setfact k:v      │  │ /branch <имя>    │  │
│  (auto-gen)      │  │ /delfact k        │  │ /switch <id>     │  │
│                  │  │                  │  │ /branches        │  │
│  messages[-N:]   │  │ facts -> вставля-│  │                  │  │
│  + summary       │  │ ются в API как   │  │ active_branch -> │  │
│                  │  │ системный блок   │  │ messages[-N:]    │  │
│                  │  │                  │  │ + facts_snapshot │  │
└──────────────────┘  └────────┬─────────┘  └────────┬─────────┘  │
                                │                     │            │
                       auto-extract после            └--- facts ---┘
                       каждого ответа LLM              snapshot при
                       (extract_facts_from_llm)        /checkpoint
```

---

## Уровень 3: Трёхуровневая память (Memory object)

```
/settask <текст>  --- WorkingMemory.current_task
/setpref k=v  ------- WorkingMemory.user_preferences
/remember k=v ------- LongTermMemory.knowledge_base
/decision <текст> --- LongTermMemory.decisions_log
/knowledge k:v  ----- LongTermMemory.knowledge_base
/memsave|load|show|stats|clear -- управление файлами памяти

Что куда попадает в API:

LongTermMemory.profile.to_system_prompt()
       │
       └---> ВСЕГДА добавляется к system message
            (кроме режима agent где заменяет полностью)

WorkingMemory.current_task + user_preferences
       │
       └---> ТОЛЬКО в agent/plan mode через _build_agent_state_vars()
            -> поле "Current State" в системном промпте агента
```

---

## Уровень 4: Задачи (Tasks)

```
/task new <описание>  ------- создаёт TaskPlan + TaskStep[]
                               -> LLM генерирует шаги
                               -> active_task_id = task_id
                               -> сохраняет сессию сразу

/task start|pause|resume ---- меняет TaskPlan.phase (FSM)
/task done|fail ------------- завершает план
/task step done|fail|note --- обновляет текущий TaskStep
/task load <id> ------------- переключает active_task_id
/task list|show|result ------ только чтение

active_task_id влияет на API:
       │
       └---> task context инжектируется в system message
            ПЕРЕД каждым API-запросом (если agent_mode выкл)
            В state_vars (если agent_mode вкл)

TaskPlan.clarifications
       │
       └---> инжектируются в _build_agent_state_vars()
            -> видны в "Current State" агентского промпта

FSM фаз задачи:
planning -> execution -> validation -> done
                      -> paused
         -> failed (из любой активной фазы)
```

---

## Уровень 5: План / Агент (Agent mode + Invariants)

```
/invariant add|del|edit|list|clear -- AgentMode.invariants[]

/plan on  ---------------------- включает FSM:
  │   plan_dialog_state:
  │   awaiting_task -> awaiting_invariants -> active -> confirming
  │
  │   В состоянии active:
  │   - system prompt ЗАМЕНЯЕТСЯ на build_plan_dialog_prompt(invariants)
  │   - каждый ответ LLM проходит validate_draft_against_invariants()
  │     (до max_retries попыток)
  │
  └-- после confirming -> создаёт TaskPlan через /task new

/plan off  --------------------- выключает FSM, сбрасывает plan_dialog_state
/plan retries <n>  ------------- AgentMode.max_retries

/plan builder  ----------------- запускает автоматическое выполнение шагов:
  │   требует: active_task_id
  │   берёт: profile + working_state_vars + invariants
  │         + step_title + step_description + previous_steps
  │   каждый шаг:
  │     1. build_builder_step_prompt(...)
  │     2. LLM вызов
  │     3. validate_draft_against_invariants() loop
  │     4. при fail -> clarification question -> retry
  │     5. при исчерпании -> _prompt_invariant_resolution:
  │           edit/remove invariant или abort
  │     6. step.result = draft -> save_task_step()
  │   после всех шагов:
  │     -> TaskPlan.result = join(step.result)
  │     -> фаза EXECUTION -> VALIDATION -> DONE

Обычный диалог с agent_mode.enabled=True (без plan FSM):
  - system prompt = build_agent_system_prompt(profile, state_vars, invariants)
  - после ответа: State Update -> WorkingMemory.user_preferences
  - валидация на каждый ответ
```

---

## Уровень 6: Проекты

```
/project new <имя>  ------- создаёт Project, active_project_id = id
/project list  ------------- список проектов профиля
/project switch <имя>  ----- меняет active_project_id
/project show  ------------- показывает Project + его TaskPlan'ы
/project add-plan  --------- добавляет active_task_id в project.plan_ids

Project влияет на API: НЕ ВЛИЯЕТ (только организационный уровень)
```

---

## Итоговая сборка API-запроса

```
[system message] =
  ┌- если agent_mode.enabled:
  │    build_agent_system_prompt(
  │      profile_text  <- LongTermMemory.profile.to_system_prompt()
  │      state_vars    <- WorkingMemory + active_task_id + clarifications
  │      invariants    <- AgentMode.invariants
  │    )
  │    (заменяет системный промпт полностью)
  │
  └- если agent_mode.disabled:
       base system_prompt (из --system-prompt или session)
       + task_context (если active_task_id)
       + profile_text  <- LongTermMemory.profile.to_system_prompt()

[messages] =
  ┌- strategy=sliding_window:  messages[-N:] + context_summary
  ├- strategy=sticky_facts:    sticky_facts block + messages[-N:]
  └- strategy=branching:       branch.facts_snapshot + branch.messages[-N:]

после ответа (если strategy=sticky_facts):
  extract_facts_from_llm() -> автообновление sticky_facts
после ответа (если agent_mode.enabled):
  State Update block -> WorkingMemory.user_preferences
```

---

## Сводная таблица зависимостей

| Настройка | Глобальность | Влияет на API | Сбрасывается при |
|---|---|---|---|
| `profile_name` | namespace для всего | нет (косвенно) | `/profile load` |
| `UserProfile` | весь профиль | всегда (system msg) | `/profile load` |
| `AgentMode.invariants` | сессия | только в agent/plan | `/profile load` |
| `active_task_id` | сессия | всегда (если задан) | `/task done/fail`, `/profile load` |
| `active_project_id` | сессия | нет | `/profile load` |
| `sticky_facts` | сессия | только при strategy=sf | `/memclear`, `/delfact` |
| `branches` | сессия | только при strategy=br | `/profile load` |
| `WorkingMemory` | файл + сессия | только в agent/plan | `/memclear working` |
| `context_summary` | сессия | только при strategy=sw | накапливается авто |
| `plan_dialog_state` | сессия | перехватывает ввод | `/plan off`, `/profile load` |
