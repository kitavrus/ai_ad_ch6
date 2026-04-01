# AI PR Review Pipeline — Руководство по воспроизведению

Это руководство описывает, как с нуля поднять автоматическое AI-ревью кода на Pull Request.

---

## Что получится в итоге

При открытии или обновлении PR в GitHub:
1. GitHub Action отправляет diff на удалённый сервер
2. Сервер запускает RAG + LLM-анализ
3. В PR автоматически появляется комментарий с ревью
4. Если в коде есть `TODO` / `FIXME` / `HACK` — Action **завершается с ошибкой** (красный крест)

---

## Часть 1 — Файлы в репозитории

### Шаг 1. Создать модуль с логикой ревью

Файл: `llm_agent_v2/pr_review.py`

Что делает:
- Принимает git diff и список изменённых файлов
- Ищет TODO/FIXME/HACK в добавленных строках (`+` в начале)
- Делает RAG-запрос к существующему `rag_index/fixed.faiss` для контекста
- Вызывает LLM через `LLMClient` с системным промптом ревьюера
- Возвращает кортеж `(review_text, blocking_issues)`

Ключевые функции:
```python
detect_blocking_issues(diff: str) -> list[str]   # ищет TODO/FIXME/HACK
run_review_sync(diff, changed_files) -> (text, blocking)
format_comment(review_text, pr_number, blocking) -> str
```

Системный промпт требует от LLM три раздела:
- `## 🐛 Potential Bugs`
- `## 🏗️ Architectural Issues`
- `## 💡 Recommendations`

### Шаг 2. Добавить endpoint в FastAPI сервер

Файл: `llm_agent_v2/web/api_server.py` — добавить в конец перед `if __name__ == "__main__":`

Новые Pydantic-модели:
```python
class PRReviewRequest(BaseModel):
    diff: str
    changed_files: list[str]
    pr_number: int
    repo: str          # "owner/repo"
    github_token: str

class PRReviewResponse(BaseModel):
    review: str
    comment_url: Optional[str] = None
    has_blocking_issues: bool = False
    blocking_issues: list[str] = []
```

Новый endpoint `POST /pr-review`:
1. Проверяет заголовок `X-Review-Token` против переменной окружения `REVIEW_SECRET`
2. Вызывает `run_review()` из `pr_review.py`
3. Постит комментарий в PR через GitHub API (`/repos/{repo}/issues/{pr}/comments`)
4. Возвращает `PRReviewResponse` с флагом `has_blocking_issues`

Добавить в раздел Config:
```python
REVIEW_SECRET = os.getenv("REVIEW_SECRET", "")
```

### Шаг 3. Создать GitHub Action

Файл: `.github/workflows/pr_review.yml`

```yaml
on:
  pull_request:
    types: [opened, synchronize, reopened]

jobs:
  ai-review:
    runs-on: ubuntu-latest
    permissions:
      pull-requests: write
    steps:
      - uses: actions/checkout@v4
        with: { fetch-depth: 0 }

      - name: Get diff and changed files
        run: |
          git diff origin/${{ github.base_ref }}...HEAD > /tmp/diff.patch
          git diff --name-only origin/${{ github.base_ref }}...HEAD > /tmp/changed_files.txt

      - name: Send to AI review server
        env:
          REVIEW_SECRET: ${{ secrets.REVIEW_SECRET }}
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          python3 - <<'EOF'
          import json, urllib.request, urllib.error, pathlib, sys, os
          diff = pathlib.Path("/tmp/diff.patch").read_text(errors="replace")
          files = [f for f in pathlib.Path("/tmp/changed_files.txt").read_text().splitlines() if f]
          if not diff.strip():
              print("Empty diff — skipping.")
              sys.exit(0)
          payload = json.dumps({
              "diff": diff, "changed_files": files,
              "pr_number": ${{ github.event.number }},
              "repo": "${{ github.repository }}",
              "github_token": os.environ["GH_TOKEN"],
          }).encode()
          req = urllib.request.Request(
              "http://<SERVER_IP>:<PORT>/pr-review",
              data=payload,
              headers={"Content-Type": "application/json",
                       "X-Review-Token": os.environ.get("REVIEW_SECRET", "")},
              method="POST",
          )
          try:
              result = json.loads(urllib.request.urlopen(req, timeout=180).read())
              print("Review posted:", result.get("comment_url"))
              if result.get("has_blocking_issues"):
                  print("❌ BLOCKING: TODO/FIXME found!", file=sys.stderr)
                  for issue in result.get("blocking_issues", []):
                      print(f"  - {issue}", file=sys.stderr)
                  sys.exit(1)
          except urllib.error.HTTPError as e:
              print(f"Error {e.code}: {e.read().decode()}", file=sys.stderr)
              sys.exit(1)
          EOF
```

Заменить `<SERVER_IP>` и `<PORT>` на реальные значения.

---

## Часть 2 — Сервер

### Шаг 4. Скопировать файлы на сервер

```bash
scp llm_agent_v2/pr_review.py debian@<SERVER_IP>:/home/debian/llm/pr_review.py
scp llm_agent_v2/web/api_server.py debian@<SERVER_IP>:/home/debian/llm/web/api_server.py
```

### Шаг 5. Сгенерировать секретный токен

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

Сохранить вывод — это `REVIEW_SECRET`.

### Шаг 6. Добавить REVIEW_SECRET в .env на сервере

```bash
echo 'REVIEW_SECRET=<сгенерированный_токен>' >> /home/debian/llm/.env
```

### Шаг 7. Перезапустить сервис

```bash
sudo systemctl restart llm-api.service
systemctl is-active llm-api.service   # должно быть: active
```

### Шаг 8. Проверить endpoint

```bash
# Должен вернуть 403
curl -X POST http://<SERVER_IP>:<PORT>/pr-review \
  -H "X-Review-Token: wrong" \
  -H "Content-Type: application/json" \
  -d '{"diff":"x","changed_files":[],"pr_number":1,"repo":"a/b","github_token":"x"}'

# Должен вернуть 200 от /health
curl http://<SERVER_IP>:<PORT>/health
```

---

## Часть 3 — GitHub

### Шаг 9. Добавить GitHub Secret

Перейти: `https://github.com/<owner>/<repo>/settings/secrets/actions/new`

| Name | Value |
|------|-------|
| `REVIEW_SECRET` | `<токен из шага 5>` |

**Через API** (если есть PAT с правом `repo`):
```python
import urllib.request, json, base64, subprocess, sys, os, tempfile

PAT = "<github_pat>"
REPO = "owner/repo"
SECRET_VALUE = "<review_secret>"

# 1. Получить публичный ключ репозитория
headers = {"Authorization": f"Bearer {PAT}", "Accept": "application/vnd.github+json",
           "X-GitHub-Api-Version": "2022-11-28"}
req = urllib.request.Request(f"https://api.github.com/repos/{REPO}/actions/secrets/public-key",
                             headers=headers)
pk = json.loads(urllib.request.urlopen(req).read())

# 2. Зашифровать секрет через PyNaCl в venv
venv = tempfile.mkdtemp()
subprocess.check_call([sys.executable, "-m", "venv", venv])
venv_py = os.path.join(venv, "bin", "python")
subprocess.check_call([venv_py, "-m", "pip", "install", "PyNaCl", "-q"])
script = f"""
import base64
import sys, glob
for p in glob.glob('{venv}/lib/python*/site-packages'): sys.path.insert(0, p)
from nacl.public import PublicKey, SealedBox
box = SealedBox(PublicKey(base64.b64decode("{pk['key']}")))
print(base64.b64encode(box.encrypt("{SECRET_VALUE}".encode())).decode())
"""
encrypted = subprocess.check_output([venv_py, "-c", script]).decode().strip()

# 3. Отправить секрет
put = urllib.request.Request(
    f"https://api.github.com/repos/{REPO}/actions/secrets/REVIEW_SECRET",
    data=json.dumps({"encrypted_value": encrypted, "key_id": pk["key_id"]}).encode(),
    headers={**headers, "Content-Type": "application/json"}, method="PUT")
r = urllib.request.urlopen(put)
print(f"Secret added: HTTP {r.status}")
```

---

## Часть 4 — Проверка

### Шаг 10. Создать тестовый PR с TODO

```bash
git checkout -b test/review-check
cat > test_todo.py << 'EOF'
def my_function():
    # TODO: implement this
    pass
EOF
git add test_todo.py
git commit -m "test: add TODO to trigger review failure"
git push origin test/review-check
```

Открыть PR через UI или API. Ожидаемый результат:
- Action **красный** (exit 1)
- В PR появился комментарий с `## ❌ Blocking Issues`
- Перечислены строки с TODO

### Шаг 11. Создать чистый PR (без TODO)

```bash
git checkout -b test/clean-review
cat > clean_feature.py << 'EOF'
def add(a: float, b: float) -> float:
    return a + b
EOF
git add clean_feature.py
git commit -m "feat: add utility function"
git push origin test/clean-review
```

Ожидаемый результат:
- Action **зелёный**
- Комментарий с `## 🤖 AI Code Review` без блокирующих проблем

---

## Итоговая структура файлов

```
.github/
  workflows/
    pr_review.yml          ← GitHub Action

llm_agent_v2/
  pr_review.py             ← бизнес-логика ревью (новый файл)
  web/
    api_server.py          ← добавлен POST /pr-review endpoint

/home/debian/llm/          ← на сервере
  .env                     ← добавлен REVIEW_SECRET=...
  pr_review.py             ← скопирован с локального
  web/
    api_server.py          ← скопирован с локального
```

---

## Переменные окружения

| Переменная | Где | Описание |
|------------|-----|----------|
| `REVIEW_SECRET` | `.env` на сервере | Shared secret для защиты endpoint |
| `REVIEW_SECRET` | GitHub Secrets | Тот же токен, передаётся из Action |
| `GITHUB_TOKEN` | Авто в Action | GitHub предоставляет автоматически |
| `API_KEY` | `.env` на сервере | Ключ LLM API |
| `BASE_URL` | `.env` на сервере | URL LLM API (OpenAI-совместимый) |
