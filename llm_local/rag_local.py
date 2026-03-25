"""Локальный RAG-пайплайн: FAISS retrieval + Ollama generation.

Запуск (интерактивный тест):
    cd /Users/igorpotema/mycode/ai_ad_ch6
    python llm_local/rag_local.py
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional

# Добавляем корень проекта в sys.path, чтобы импортировать llm_agent
sys.path.insert(0, str(Path(__file__).parent.parent))

import ollama
from dotenv import find_dotenv, load_dotenv
from openai import OpenAI

load_dotenv(find_dotenv(usecwd=True))

DEFAULT_LOCAL_MODEL = "qwen3:14b"
DEFAULT_CLOUD_MODEL = "inception/mercury-coder"
DEFAULT_TEMPERATURE = 0.3
DEFAULT_TOP_K = 3
DEFAULT_INDEX_DIR = str(Path(__file__).parent.parent / "llm_agent" / "rag_index")


class LocalRAGPipeline:
    """RAG-пайплайн с локальной генерацией через Ollama и облачной через OpenAI.

    Retrieval всегда локальный (FAISS).
    Embeddings для запроса — через OpenAI API (индекс построен на text-embedding-3-small).
    Генерация — Ollama (локально) или OpenAI (облако).
    """

    def __init__(
        self,
        local_model: str = DEFAULT_LOCAL_MODEL,
        cloud_model: str = DEFAULT_CLOUD_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        top_k: int = DEFAULT_TOP_K,
        index_dir: Optional[str] = None,
        relevance_threshold: float = 0.0,
        openai_api_key: Optional[str] = None,
        openai_base_url: Optional[str] = None,
    ):
        self.local_model = local_model
        self.cloud_model = cloud_model
        self.temperature = temperature
        self.top_k = top_k
        self.index_dir = index_dir or DEFAULT_INDEX_DIR
        self.relevance_threshold = relevance_threshold

        # OpenAI credentials (для embeddings и облачной генерации)
        self._api_key = (
            openai_api_key
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("API_KEY")
        )
        self._base_url = openai_base_url or os.getenv("OPENAI_BASE_URL")

        self._retriever = None  # ленивая инициализация
        self._ollama_client = ollama.Client()

    def _get_retriever(self):
        """Возвращает кэшированный RAGRetriever."""
        if self._retriever is None:
            from llm_agent.rag.retriever import RAGRetriever
            self._retriever = RAGRetriever(
                index_dir=self.index_dir,
                relevance_threshold=self.relevance_threshold,
            )
        return self._retriever

    def _get_openai_client(self) -> OpenAI:
        """Создаёт OpenAI-клиент. Выбрасывает RuntimeError если ключ не задан."""
        if not self._api_key:
            raise RuntimeError(
                "OPENAI_API_KEY / API_KEY не задан. Облачные режимы недоступны."
            )
        kwargs: dict = {"api_key": self._api_key}
        if self._base_url:
            kwargs["base_url"] = self._base_url
        return OpenAI(**kwargs)

    def _empty_result(self, reason: str) -> dict:
        return {
            "answer": f"(skipped: {reason})",
            "search_time": 0.0,
            "generate_time": 0.0,
            "total_time": 0.0,
            "char_count": 0,
            "rag_used": False,
        }

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: Optional[int] = None):
        """Поиск в FAISS-индексе. Возвращает (results, elapsed_seconds)."""
        k = top_k or self.top_k
        t0 = time.perf_counter()
        retriever = self._get_retriever()
        results = retriever.search_with_scores(query, strategy="structure", top_k=k)
        elapsed = time.perf_counter() - t0
        return results, elapsed

    # ------------------------------------------------------------------
    # Local generation (Ollama)
    # ------------------------------------------------------------------

    def ask_local_norag(self, query: str) -> dict:
        """Ответ локальной модели без RAG-контекста."""
        t0 = time.perf_counter()
        resp = self._ollama_client.generate(
            model=self.local_model,
            prompt=query,
            options={"temperature": self.temperature},
        )
        generate_time = time.perf_counter() - t0
        answer = resp["response"]
        return {
            "answer": answer,
            "search_time": 0.0,
            "generate_time": generate_time,
            "total_time": generate_time,
            "char_count": len(answer),
            "rag_used": False,
        }

    def ask_local(self, query: str) -> dict:
        """Ответ локальной модели с RAG-контекстом (FAISS + Ollama)."""
        from llm_agent.chatbot.context import build_rag_system_addition

        results, search_time = self.search(query)
        if not results:
            return {**self.ask_local_norag(query), "search_time": search_time, "rag_used": False}

        system_text = build_rag_system_addition(results)
        prompt = f"{system_text}\n\nВопрос: {query}"

        t0 = time.perf_counter()
        resp = self._ollama_client.generate(
            model=self.local_model,
            prompt=prompt,
            options={"temperature": self.temperature},
        )
        generate_time = time.perf_counter() - t0
        answer = resp["response"]

        return {
            "answer": answer,
            "search_time": search_time,
            "generate_time": generate_time,
            "total_time": search_time + generate_time,
            "char_count": len(answer),
            "rag_used": True,
        }

    # ------------------------------------------------------------------
    # Cloud generation (OpenAI)
    # ------------------------------------------------------------------

    def ask_cloud_norag(self, query: str) -> dict:
        """Ответ облачной модели без RAG-контекста."""
        try:
            client = self._get_openai_client()
        except RuntimeError as e:
            return self._empty_result(str(e))

        t0 = time.perf_counter()
        response = client.chat.completions.create(
            model=self.cloud_model,
            messages=[{"role": "user", "content": query}],
            max_tokens=512,
            temperature=self.temperature,
        )
        generate_time = time.perf_counter() - t0
        answer = response.choices[0].message.content or ""

        return {
            "answer": answer,
            "search_time": 0.0,
            "generate_time": generate_time,
            "total_time": generate_time,
            "char_count": len(answer),
            "rag_used": False,
        }

    def ask_cloud(self, query: str) -> dict:
        """Ответ облачной модели с RAG-контекстом (FAISS + OpenAI)."""
        try:
            client = self._get_openai_client()
        except RuntimeError as e:
            return self._empty_result(str(e))

        from llm_agent.chatbot.context import build_rag_system_addition

        results, search_time = self.search(query)
        if not results:
            return {**self.ask_cloud_norag(query), "search_time": search_time, "rag_used": False}

        system_text = build_rag_system_addition(results)

        t0 = time.perf_counter()
        response = client.chat.completions.create(
            model=self.cloud_model,
            messages=[
                {"role": "system", "content": system_text},
                {"role": "user", "content": query},
            ],
            max_tokens=512,
            temperature=self.temperature,
        )
        generate_time = time.perf_counter() - t0
        answer = response.choices[0].message.content or ""

        return {
            "answer": answer,
            "search_time": search_time,
            "generate_time": generate_time,
            "total_time": search_time + generate_time,
            "char_count": len(answer),
            "rag_used": True,
        }


# ----------------------------------------------------------------------
# Интерактивный тест
# ----------------------------------------------------------------------

TEST_QUESTIONS = [
    "Как устроена трёхуровневая система памяти?",
    "Какие стратегии управления контекстом поддерживаются?",
    "Что происходит при нарушении инварианта в plan builder?",
]

if __name__ == "__main__":
    print(f"Инициализация LocalRAGPipeline (модель: {DEFAULT_LOCAL_MODEL})")
    pipeline = LocalRAGPipeline()
    has_cloud = bool(os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY"))
    print(f"Облачная генерация: {'доступна' if has_cloud else 'НЕДОСТУПНА (нет API_KEY)'}")
    print("=" * 60)

    for i, q in enumerate(TEST_QUESTIONS, 1):
        print(f"\n[Вопрос {i}] {q}")
        print("-" * 60)

        # Локально без RAG
        r_ln = pipeline.ask_local_norag(q)
        print(f"LOCAL_NORAG ({r_ln['total_time']:.1f}s, {r_ln['char_count']} симв.):")
        print(r_ln["answer"])

        # Локально с RAG
        r_lr = pipeline.ask_local(q)
        rag_tag = "[RAG]" if r_lr["rag_used"] else "[нет чанков]"
        print(f"\nLOCAL_RAG {rag_tag} ({r_lr['total_time']:.1f}s = {r_lr['search_time']:.1f}s поиск + {r_lr['generate_time']:.1f}s генерация, {r_lr['char_count']} симв.):")
        print(r_lr["answer"])

        if has_cloud:
            # Облако с RAG
            r_cr = pipeline.ask_cloud(q)
            rag_tag = "[RAG]" if r_cr["rag_used"] else "[нет чанков]"
            print(f"\nCLOUD_RAG {rag_tag} ({r_cr['total_time']:.1f}s, {r_cr['char_count']} симв.):")
            print(r_cr["answer"])

        print("=" * 60)
