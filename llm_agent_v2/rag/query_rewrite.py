"""LLM-based query rewriter for improving RAG retrieval."""

import os
from typing import List, Optional


class QueryRewriter:
    """Rewrite a user query to improve embedding-based retrieval."""

    _SINGLE_PROMPT = (
        "Rewrite the following search query to improve retrieval from a technical "
        "documentation knowledge base. Return ONLY the rewritten query, nothing else.\n\n"
        "Query: {query}"
    )

    _MULTI_PROMPT = (
        "Generate {n} distinct search query variants for the following query to improve "
        "retrieval from a technical documentation knowledge base. "
        "Return ONLY the variants, one per line, no numbering or extra text.\n\n"
        "Query: {query}"
    )

    def __init__(self, client=None, model: Optional[str] = None):
        self._client = client
        self._model = model or os.getenv("DEFAULT_MODEL", "inception/mercury-coder")

    def _get_client(self):
        if self._client is not None:
            return self._client
        from openai import OpenAI
        from dotenv import find_dotenv, load_dotenv
        load_dotenv(find_dotenv(usecwd=True))
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        kwargs: dict = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)
        return self._client

    def rewrite(self, query: str) -> str:
        """Return a single rewritten query; falls back to original on any error."""
        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "user", "content": self._SINGLE_PROMPT.format(query=query)}
                ],
                max_tokens=128,
                temperature=0.3,
            )
            rewritten = response.choices[0].message.content or ""
            rewritten = rewritten.strip()
            return rewritten if rewritten else query
        except Exception:
            return query

    def rewrite_multi(self, query: str, n: int = 3) -> List[str]:
        """Return N query variants; falls back to [original] on any error."""
        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "user",
                        "content": self._MULTI_PROMPT.format(query=query, n=n),
                    }
                ],
                max_tokens=256,
                temperature=0.5,
            )
            raw = response.choices[0].message.content or ""
            variants = [line.strip() for line in raw.splitlines() if line.strip()]
            # deduplicate while preserving order
            seen: set = set()
            unique: List[str] = []
            for v in variants:
                if v not in seen:
                    seen.add(v)
                    unique.append(v)
            return unique if unique else [query]
        except Exception:
            return [query]
