from __future__ import annotations

"""Small provider adapter for generation, RAGAS judging, and enrichment."""

import json
import os
from typing import Any

from config import (
    COHERE_API_KEY,
    EMBEDDING_MODEL,
    OLLAMA_API_KEY,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OPENAI_API_KEY,
)


def _valid_openai_key() -> bool:
    return (
        os.getenv("ENABLE_OPENAI_FALLBACK") == "1"
        and OPENAI_API_KEY.startswith("sk-")
        and len(OPENAI_API_KEY) > 20
    )


def _valid_ollama_key() -> bool:
    return bool(OLLAMA_API_KEY and len(OLLAMA_API_KEY) > 10)


def _valid_cohere_key() -> bool:
    return os.getenv("ENABLE_COHERE_FALLBACK") == "1" and bool(COHERE_API_KEY and len(COHERE_API_KEY) > 20)


def chat(system: str, user: str, json_mode: bool = False, max_tokens: int = 500) -> str:
    """Generate text using Ollama Cloud/OpenAI-compatible first, then OpenAI, then Cohere."""
    errors: list[str] = []
    if _valid_ollama_key():
        try:
            from openai import OpenAI

            # Ollama Cloud lộ endpoint OpenAI-compatible tại /v1 — dùng thẳng
            # OpenAI client với base_url khác.
            prompt_user = user
            if json_mode:
                prompt_user += "\n\nChỉ trả về JSON hợp lệ, không thêm markdown/code fence."
            resp = OpenAI(api_key=OLLAMA_API_KEY, base_url=OLLAMA_BASE_URL).chat.completions.create(
                model=OLLAMA_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt_user},
                ],
                temperature=0,
                max_tokens=max_tokens,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            errors.append(f"Ollama Cloud failed: {e}")

    if _valid_openai_key():
        try:
            from openai import OpenAI

            kwargs: dict[str, Any] = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0,
                "max_tokens": max_tokens,
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            resp = OpenAI(api_key=OPENAI_API_KEY).chat.completions.create(**kwargs)
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            errors.append(f"OpenAI failed: {e}")

    if _valid_cohere_key():
        try:
            import cohere

            prompt = f"{system}\n\n{user}"
            if json_mode:
                prompt += "\n\nChỉ trả về JSON hợp lệ, không thêm markdown."
            resp = cohere.Client(COHERE_API_KEY).chat(
                model="command-r-plus",
                message=prompt,
                temperature=0,
                max_tokens=max_tokens,
            )
            return (getattr(resp, "text", "") or "").strip()
        except Exception as e:
            errors.append(f"Cohere failed: {e}")

    raise RuntimeError("No working LLM provider configured" + (": " + " | ".join(errors) if errors else ""))


def chat_json(system: str, user: str, max_tokens: int = 500) -> dict:
    text = chat(system, user, json_mode=True, max_tokens=max_tokens)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise


def get_ragas_llm():
    if _valid_ollama_key():
        from langchain_openai import ChatOpenAI
        from ragas.llms import LangchainLLMWrapper

        return LangchainLLMWrapper(ChatOpenAI(
            model=OLLAMA_MODEL,
            temperature=0,
            api_key=OLLAMA_API_KEY,
            base_url=OLLAMA_BASE_URL,
        ))

    if _valid_openai_key():
        from langchain_openai import ChatOpenAI
        from ragas.llms import LangchainLLMWrapper

        return LangchainLLMWrapper(ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=OPENAI_API_KEY,
        ))

    if _valid_cohere_key():
        from langchain_community.chat_models import ChatCohere
        from ragas.llms import LangchainLLMWrapper

        # ⚠️ Last resort — đã xác nhận bằng chạy thật: ChatCohere (langchain_community,
        # deprecated) luôn tự thêm 'temperature' vào request ở _default_params, còn
        # ragas 0.1.x cũng tự truyền 'temperature' riêng vào generate() → luôn
        # TypeError "got multiple values for keyword argument 'temperature'" trên
        # MỌI câu hỏi (verified: 80/80 job fail). Không sửa được từ phía config
        # (lỗi nằm trong logic nội bộ của ChatCohere), chỉ dùng khi không có
        # OpenAI lẫn Ollama Cloud key.
        return LangchainLLMWrapper(ChatCohere(
            model="command-r-plus",
            cohere_api_key=COHERE_API_KEY,
        ))

    return None


def get_ragas_embeddings():
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception:
        return None
