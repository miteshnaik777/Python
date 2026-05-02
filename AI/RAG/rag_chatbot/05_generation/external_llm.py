"""
05_generation/external_llm.py
──────────────────────────────
LLM via OpenAI-compatible APIs (OpenAI, Groq, Ollama) or Google Gemini.

Set LLM_PROVIDER and the corresponding API key in `.env` (see EXTERNAL_LLM_SETUP.md).

Usage:
    from generation.external_llm import ExternalLLM

    llm = ExternalLLM()
    answer = llm.generate_messages(messages)  # messages = [{"role":"system","content":"..."}, ...]
"""

import re
import sys
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import cfg

log = logging.getLogger(__name__)


def _get_openai_client() -> Tuple[Any, str]:
    """Build OpenAI client for OpenAI-compatible providers."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Install the OpenAI package: pip install openai")
    provider = (cfg.LLM_PROVIDER or "").strip().lower() or "openai"
    base_url = None
    api_key = ""
    model = cfg.EXTERNAL_LLM_MODEL or "gpt-4o-mini"

    if provider == "groq":
        base_url = "https://api.groq.com/openai/v1"
        api_key = cfg.GROQ_API_KEY or cfg.EXTERNAL_LLM_API_KEY
        model = cfg.EXTERNAL_LLM_MODEL or "llama-3.3-70b-versatile"
    elif provider == "openai":
        base_url = "https://api.openai.com/v1"
        api_key = cfg.OPENAI_API_KEY or cfg.EXTERNAL_LLM_API_KEY
        model = cfg.EXTERNAL_LLM_MODEL or "gpt-4o-mini"
    elif provider == "ollama":
        base_url = "http://localhost:11434/v1"
        api_key = "ollama"
        model = cfg.EXTERNAL_LLM_MODEL or "llama3.2"
    elif provider == "openai_compatible":
        base_url = cfg.EXTERNAL_LLM_BASE_URL
        api_key = cfg.EXTERNAL_LLM_API_KEY or "not-set"
        model = cfg.EXTERNAL_LLM_MODEL or "llama-3.3-70b-versatile"
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{provider}'. Use one of: openai, gemini, groq, ollama, openai_compatible"
        )

    if not api_key and provider != "ollama":
        raise ValueError(
            f"Set the API key in .env for {provider}: OPENAI_API_KEY, GEMINI_API_KEY, GROQ_API_KEY, or EXTERNAL_LLM_API_KEY"
        )
    kwargs = {"api_key": api_key or "ollama"}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs), model


def _get_gemini_model() -> Tuple[Any, str]:
    """Build Gemini model. Returns (model, model_name)."""
    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError("Install the Google Generative AI package: pip install google-generativeai")
    api_key = cfg.GEMINI_API_KEY or cfg.EXTERNAL_LLM_API_KEY
    if not api_key:
        raise ValueError("Set GEMINI_API_KEY in .env. Get a key at https://aistudio.google.com/app/apikey")
    genai.configure(api_key=api_key)
    # Use a Gemini model name. Old "gemini-1.5-flash" often 404s; use 2.0-flash or set EXTERNAL_LLM_MODEL=gemini-pro.
    raw = (cfg.EXTERNAL_LLM_MODEL or "").strip()
    model_name = raw if raw.lower().startswith("gemini") else "gemini-2.0-flash"
    model = genai.GenerativeModel(model_name)
    return model, model_name


class ExternalLLM:
    """
    Call OpenAI-compatible APIs (OpenAI, Groq, Ollama) or Google Gemini.
    """

    def __init__(self, model: str = None):
        provider = (cfg.LLM_PROVIDER or "").strip().lower() or "openai"
        self._provider = provider
        self._gemini_model = None
        self._openai_client = None
        if provider == "gemini":
            self._gemini_model, self._model = _get_gemini_model()
            self.model = model or self._model
        else:
            self._openai_client, self._model = _get_openai_client()
            self.model = model or self._model
        log.info("ExternalLLM: provider=%s, model=%s", provider, self.model)

    def generate_messages(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = None,
        temperature: float = None,
    ) -> str:
        """
        Send chat messages and return the assistant reply.
        """
        max_tokens = max_tokens if max_tokens is not None else cfg.MAX_NEW_TOKENS
        temperature = temperature if temperature is not None else cfg.TEMPERATURE
        if self._provider == "gemini":
            return self._generate_gemini(messages, max_tokens, temperature)
        return self._generate_openai(messages, max_tokens, temperature)

    def _generate_gemini(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Use Gemini API: combine system + user into content for generate_content."""
        parts = []
        for m in messages:
            role = (m.get("role") or "").strip()
            content = (m.get("content") or "").strip()
            if not content:
                continue
            if role == "system":
                parts.append(f"Instructions: {content}\n\n")
            else:
                parts.append(content)
        prompt = "\n".join(parts).strip() or "No content."
        last_error = None
        for attempt in range(2):
            try:
                response = self._gemini_model.generate_content(
                    prompt,
                    generation_config={
                        "max_output_tokens": max_tokens,
                        "temperature": temperature,
                    },
                )
                if response and response.text:
                    return response.text.strip()
                return ""
            except Exception as e:
                last_error = e
                err_str = str(e)
                err_lower = err_str.lower()
                if "api_key" in err_lower or "403" in err_lower or "invalid" in err_lower:
                    raise RuntimeError(
                        "Invalid or missing GEMINI_API_KEY. Get a key at https://aistudio.google.com/app/apikey"
                    ) from e
                if "429" in err_str or "quota" in err_lower or "rate" in err_lower:
                    if attempt == 0:
                        match = re.search(r"retry in (\d+(?:\.\d+)?)\s*s", err_str, re.I)
                        delay = float(match.group(1)) + 1 if match else 18
                        log.info("Gemini 429 quota/rate limit — waiting %.0fs then retrying once", delay)
                        time.sleep(delay)
                        continue
                    raise RuntimeError(
                        "Gemini free-tier quota exceeded. Wait a few minutes and try again, or use another provider: "
                        "in .env set LLM_PROVIDER=groq and GROQ_API_KEY=... (free at console.groq.com), "
                        "or LLM_PROVIDER=openai and OPENAI_API_KEY=... . See EXTERNAL_LLM_SETUP.md in the project folder."
                    ) from e
                raise RuntimeError(f"Gemini API error: {e}") from e
        if last_error:
            raise RuntimeError(f"Gemini API error: {last_error}") from last_error
        return ""

    def _generate_openai(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Use OpenAI-compatible chat completions."""
        try:
            resp = self._openai_client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as e:
            err = str(e).lower()
            if "api_key" in err or "auth" in err or "401" in err:
                raise RuntimeError(
                    f"Invalid or missing API key for {cfg.LLM_PROVIDER}. "
                    "Check OPENAI_API_KEY, GROQ_API_KEY, or EXTERNAL_LLM_API_KEY in .env."
                ) from e
            if "connection" in err or ("refused" in err and "ollama" in (cfg.LLM_PROVIDER or "")):
                raise RuntimeError(
                    "Ollama is not running. Start it with: ollama serve (and ollama pull llama3.2)."
                ) from e
            raise RuntimeError(f"LLM API error: {e}") from e
        choice = resp.choices[0] if resp.choices else None
        if not choice or not getattr(choice, "message", None):
            return ""
        return (choice.message.content or "").strip()
