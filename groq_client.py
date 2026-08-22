"""
Groq API Client Wrapper Module
===============================
This module provides a clean, error-handled interface to the Groq API.
It wraps the official `groq` SDK and adds:
  - Automatic API key discovery (Streamlit secrets → env var → .env file)
  - Configurable model selection with sensible defaults
  - Structured error types for common failure modes
  - Token/usage extraction from responses
  - Optional streaming support

Usage:
    from groq_client import GroqClient, build_messages

    client = GroqClient(api_key="...")
    messages = build_messages("You are a helpful assistant.", user_query="Hi!")
    response = client.chat(messages)
    print(response.content)
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# Dependencies: groq SDK
# -----------------------------------------------------------------------------
# We import the Groq client lazily in the class so the module can be imported
# even without the SDK installed (useful for static analysis / linting).
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

# ═══════════════════════════════════════════════════════════════════════════════
# SINGLE FIXED MODEL LOCKED FOR ALL REQUESTS (no user choice exposed in UI).
#
# Picked openai/gpt-oss-120b because:
#   • Launched Aug 2025 on Groq — definitely NOT decommissioned (Aug 2026 date)
#   • Groq's current flagship open-weight model (131k ctx window, ~500 t/s)
#   • Official replacement suggested by Groq for retired llama-3.3/3.1 lineups
#   • Supports tool use, browser search, code execution, reasoning OOTB
#
# If Groq ever retires THIS model, update the string below to a new model from
# https://console.groq.com/docs/models — that's the only change required.
# ═══════════════════════════════════════════════════════════════════════════════
DEFAULT_MODEL: str = "openai/gpt-oss-120b"

# The UI no longer lets users pick a model, but we keep this list with exactly
# ONE entry so existing code paths (index-in-list lookups, etc.) don't break.
COMMON_GROQ_MODELS: List[str] = [
    DEFAULT_MODEL,
]


# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------

@dataclass
class ChatUsage:
    """
    Token usage information extracted from a Groq API response.

    Fields match the Groq API's `usage` object shape exactly.
    """
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    prompt_time: float = 0.0
    completion_time: float = 0.0
    total_time: float = 0.0


@dataclass
class ChatResponse:
    """
    Normalized response object returned by GroqClient.chat().
    """
    content: str
    model: str
    role: str = "assistant"
    usage: ChatUsage = field(default_factory=ChatUsage)
    raw: Any = None  # The original Groq API response object, for debugging.


# -----------------------------------------------------------------------------
# Exceptions
# -----------------------------------------------------------------------------

class GroqClientError(Exception):
    """Base class for all Groq client errors."""
    pass


class GroqAuthenticationError(GroqClientError):
    """Raised when the API key is missing, empty, or invalid."""
    pass


class GroqRateLimitError(GroqClientError):
    """Raised when the Groq API rate limit is hit."""
    pass


class GroqServiceError(GroqClientError):
    """Raised when Groq returns a 5xx / service-unavailable response."""
    pass


class GroqValidationError(GroqClientError):
    """Raised when the request payload is malformed (4xx)."""
    pass


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def build_messages(
    system_prompt: str,
    user_query: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    """
    Build the messages array required by the Groq Chat Completions API.

    Args:
        system_prompt:    The system / persona prompt (set the role's behavior).
        user_query:       The current user input string.
        conversation_history: Optional list of prior {"role": ..., "content": ...}
                              dicts (assistant + user turns, WITHOUT system).

    Returns:
        A list ready to pass to GroqClient.chat().
    """
    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

    if conversation_history:
        # Only include user/assistant turns; never duplicate a system prompt.
        for turn in conversation_history:
            if turn.get("role") in ("user", "assistant"):
                messages.append({
                    "role": turn["role"],
                    "content": str(turn.get("content", "")),
                })

    messages.append({"role": "user", "content": user_query})
    return messages


# -----------------------------------------------------------------------------
# Main Client
# -----------------------------------------------------------------------------

class GroqClient:
    """
    Lightweight wrapper around the Groq SDK Chat Completions endpoint.

    Example:
        >>> client = GroqClient(api_key="gsk_...")
        >>> msgs = build_messages("You are helpful.", "What is 2+2?")
        >>> resp = client.chat(msgs, temperature=0.2)
        >>> print(resp.content)
        4
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = DEFAULT_MODEL,
        default_temperature: float = 0.7,
        default_max_tokens: int = 4096,
    ) -> None:
        """
        Initialize the GroqClient.

        If `api_key` is None, the client tries to find one in this order:
        1. Streamlit secrets (st.secrets["GROQ_API_KEY"]) — if streamlit imported
        2. Environment variable GROQ_API_KEY
        3. .env file in the working directory (via python-dotenv)
        """
        self.default_model = default_model
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens

        # ════════════════════════════════════════════════════════════════════
        # BULLETPROOF GUARD (post-assignment override):
        # Even if a caller (or a stale @st.cache_resource cached instance)
        # passes the WRONG default_model (e.g. a decommissioned one), ALWAYS
        # force self.default_model back to the source-of-truth DEFAULT_MODEL
        # constant. No exceptions.
        #
        # This prevents the "llama-3.1-70b-versatile cached forever" bug
        # where @st.cache_resource keys on constructor args and keeps a
        # client built from a stale session_state.model from the first run.
        # ════════════════════════════════════════════════════════════════════
        self.default_model = DEFAULT_MODEL

        self._api_key = self._resolve_api_key(api_key)
        self._client = self._create_sdk_client(self._api_key)

    # -----------------------------------------------------------------
    # Internal: model-name gatekeeper
    # -----------------------------------------------------------------

    @staticmethod
    def _sanitize_model(model_arg: Optional[str]) -> str:
        """
        FINAL FIREWALL before any Groq SDK call.

        Regardless of what the caller passed (or what came from a cached
        client default), the returned model is ALWAYS the source-of-truth
        DEFAULT_MODEL constant.

        Why this exists:
          - @st.cache_resource can cache GroqClient() instances built with
            stale constructor args (e.g. default_model="llama-3.1-70b-versatile"
            from before code was patched)
          - Callers may accidentally pass session_state.model before the
            unconditional-overwrite init runs
          - ANY string containing "llama-3.1", "llama3.1", "llama-3.3", or
            containing "decommissioned" substrings is *guaranteed* wrong

        This method eradicates all of those vectors in one place.
        """
        # Future-proof blacklist — if Groq retires more model families,
        # add their prefixes here and they'll be rejected at the gate:
        _DECOMMISSIONED_HINTS = (
            "llama-3.1", "llama3.1",
            "llama-3.3", "llama3.3",
            "llama-3.2", "llama3.2",
            "gemma-7b",
            "mixtral-8x7b",
        )
        m = (model_arg or "").strip()
        # If the caller passed a blacklisted model — ignore it completely
        if any(hint.lower() in m.lower() for hint in _DECOMMISSIONED_HINTS):
            return DEFAULT_MODEL
        # If the caller passed something that doesn't match our locked model
        # exactly — ignore it too. We only talk to one model, ever.
        if m != DEFAULT_MODEL:
            return DEFAULT_MODEL
        return DEFAULT_MODEL  # redundancy — guarantees the return value

    # -----------------------------------------------------------------
    # Key resolution
    # -----------------------------------------------------------------

    @staticmethod
    def _resolve_api_key(explicit_key: Optional[str]) -> str:
        """
        Resolve the API key with the documented fallback chain.

        Also normalizes the key: users commonly paste keys wrapped in extra
        quotes (e.g. '"gsk_abc..."' from a copy-paste of TOML content) which
        Groq rejects with a cryptic 401. We strip any surrounding whitespace
        AND any paired single / double quotes from every candidate key.
        """

        def _norm(raw: Any) -> str:
            """Normalize a raw key value: stringify, strip whitespace + quotes."""
            s = str(raw).strip()
            # Strip surrounding double-quote pairs
            if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
                s = s[1:-1]
            # Strip surrounding single-quote pairs
            if len(s) >= 2 and s[0] == "'" and s[-1] == "'":
                s = s[1:-1]
            return s.strip()

        # --- 0. Explicit key passed directly to GroqClient(api_key=...) ----
        if explicit_key:
            normed = _norm(explicit_key)
            if normed:
                return normed

        # --- 1. Streamlit secrets (if running inside Streamlit) -------------
        try:
            import streamlit as st  # type: ignore
            if "GROQ_API_KEY" in st.secrets:
                normed = _norm(st.secrets["GROQ_API_KEY"])
                if normed:
                    return normed
        except Exception:
            # Streamlit not installed, secrets malformed, or file missing.
            # Silently fall through to the next candidate — this is expected
            # when running outside a Streamlit environment.
            pass

        # --- 2. Environment variable (after loading .env to populate env) --
        try:
            load_dotenv()
        except Exception:
            pass
        env_raw = os.environ.get("GROQ_API_KEY", "")
        if env_raw:
            normed = _norm(env_raw)
            if normed:
                return normed

        # --- All candidates exhausted --------------------------------------
        raise GroqAuthenticationError(
            "GROQ_API_KEY not found. Please provide it in one of the following:\n"
            "  • Pass api_key= to GroqClient()\n"
            "  • Set GROQ_API_KEY in Streamlit secrets (local: .streamlit/secrets.toml,\n"
            "    deployed: Streamlit Community Cloud → App Settings → Secrets)\n"
            "  • Set the GROQ_API_KEY environment variable or a .env file"
        )

    @staticmethod
    def _create_sdk_client(api_key: str):
        """Instantiate the Groq SDK client or raise a friendly error."""
        try:
            from groq import Groq
        except ImportError as exc:
            raise GroqClientError(
                "The 'groq' package is not installed. Install it with:\n"
                "    pip install groq\n"
                "Or: pip install -r requirements.txt"
            ) from exc

        try:
            return Groq(api_key=api_key)
        except Exception as exc:
            raise GroqAuthenticationError(
                f"Failed to initialize Groq client: {exc}"
            ) from exc

    # -----------------------------------------------------------------
    # Chat completion (non-streaming)
    # -----------------------------------------------------------------

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: float = 1.0,
        **extra_kwargs,
    ) -> ChatResponse:
        """
        Call the Groq chat completions endpoint (non-streaming).

        For streaming, use chat_stream() — it is a dedicated method that
        yields text deltas as they arrive from Groq.

        Args:
            messages:     Standard [{role, content}, ...] list.
            model:        Groq model name; overrides the client default.
            temperature:  Sampling temperature (0.0 = deterministic, 2.0 = wild).
            max_tokens:   Maximum output tokens.
            top_p:        Nucleus sampling parameter.
            extra_kwargs: Any other keyword args forwarded to the Groq SDK.

        Returns:
            A ChatResponse instance with the full content + usage metadata.
        """
        try:
            raw_response = self._client.chat.completions.create(
                messages=messages,
                # ══════════════════════════════════════════════════════════════
                # FINAL GATEKEEPER: _sanitize_model() returns DEFAULT_MODEL
                # unconditionally. NO caller input (cached default, stale
                # session state, malicious string) can bypass this.
                # ══════════════════════════════════════════════════════════════
                model=self._sanitize_model(model),
                temperature=(
                    self.default_temperature if temperature is None else temperature
                ),
                max_tokens=(
                    self.default_max_tokens if max_tokens is None else max_tokens
                ),
                top_p=top_p,
                stream=False,
                **extra_kwargs,
            )
        except Exception as exc:
            self._raise_wrapped_error(exc)

        return self._extract_response(raw_response)

    # -----------------------------------------------------------------
    # Chat completion (streaming)
    # -----------------------------------------------------------------

    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: float = 1.0,
        **extra_kwargs,
    ) -> Iterator[str]:
        """
        Generator variant of chat() that yields text chunks as they arrive.

        Yields:
            String text deltas from the assistant response.
        """
        try:
            stream = self._client.chat.completions.create(
                messages=messages,
                # ══════════════════════════════════════════════════════════════
                # FINAL GATEKEEPER: _sanitize_model() returns DEFAULT_MODEL
                # unconditionally. NO caller input (cached default, stale
                # session state, malicious string) can bypass this.
                # ══════════════════════════════════════════════════════════════
                model=self._sanitize_model(model),
                temperature=(
                    self.default_temperature if temperature is None else temperature
                ),
                max_tokens=(
                    self.default_max_tokens if max_tokens is None else max_tokens
                ),
                top_p=top_p,
                stream=True,
                **extra_kwargs,
            )
            for chunk in stream:
                delta = ""
                try:
                    if chunk.choices and chunk.choices[0].delta:
                        delta = chunk.choices[0].delta.content or ""
                except (AttributeError, IndexError):
                    pass
                if delta:
                    yield delta
        except Exception as exc:
            self._raise_wrapped_error(exc)

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _extract_response(raw_response) -> ChatResponse:
        """Convert the Groq SDK response object into a ChatResponse."""
        try:
            choice = raw_response.choices[0]
            content = choice.message.content or ""
            role = choice.message.role or "assistant"
        except (AttributeError, IndexError) as exc:
            raise GroqClientError(
                f"Unexpected response format from Groq API: {exc}"
            ) from exc

        usage = ChatUsage()
        try:
            u = raw_response.usage
            usage = ChatUsage(
                prompt_tokens=getattr(u, "prompt_tokens", 0),
                completion_tokens=getattr(u, "completion_tokens", 0),
                total_tokens=getattr(u, "total_tokens", 0),
                prompt_time=getattr(u, "prompt_time", 0.0),
                completion_time=getattr(u, "completion_time", 0.0),
                total_time=getattr(u, "total_time", 0.0),
            )
        except Exception:
            # Usage field missing or changed shape — proceed without it.
            pass

        return ChatResponse(
            content=content,
            model=getattr(raw_response, "model", ""),
            role=role,
            usage=usage,
            raw=raw_response,
        )

    @staticmethod
    def _raise_wrapped_error(original_exc: Exception) -> None:
        """
        Translate Groq SDK / HTTP errors into our typed exception hierarchy
        so UI code can catch and display them consistently.
        """
        msg = str(original_exc).lower()

        # Authentication / authorization issues
        if any(k in msg for k in ("authentication", "unauthorized", "401", "invalid api key", "api key not found")):
            raise GroqAuthenticationError(
                "🔐 Authentication failed. Double-check your GROQ_API_KEY is valid and "
                "is correctly configured in Streamlit secrets / environment variables.\n\n"
                f"Original error: {original_exc}"
            ) from original_exc

        # Rate limit / 429
        if "429" in msg or "rate limit" in msg or "too many requests" in msg:
            raise GroqRateLimitError(
                "⏳ Groq rate limit reached. Please wait a few moments and try again, "
                "or consider upgrading your Groq plan for higher limits.\n\n"
                f"Original error: {original_exc}"
            ) from original_exc

        # Server errors / 5xx
        if any(k in msg for k in ("500", "502", "503", "504", "service unavailable", "internal server", "bad gateway")):
            raise GroqServiceError(
                "⚠️ Groq service is currently unavailable. This is a temporary issue "
                "on their end — please retry in a moment.\n\n"
                f"Original error: {original_exc}"
            ) from original_exc

        # Model decommissioned / deprecated (Groq returns "model_decommissioned" code)
        if "decommissioned" in msg or "model_decommissioned" in msg:
            raise GroqValidationError(
                "🛑 **Model no longer available.** Groq has decommissioned the model "
                "currently hardcoded in the app.\n\n"
                "👉 Fix: Edit `DEFAULT_MODEL` in `groq_client.py` (around line 51) and "
                "set it to a currently-available model from Groq's model list, then "
                "fully restart the Streamlit server (Ctrl+C + `streamlit run app.py`).\n\n"
                "Groq models: https://console.groq.com/docs/models\n"
                "Groq deprecations: https://console.groq.com/docs/deprecations\n\n"
                f"Original error: {original_exc}"
            ) from original_exc

        # 400 / bad request / validation
        if any(k in msg for k in ("400", "404", "bad request", "validation", "invalid")):
            raise GroqValidationError(
                "❌ Invalid request. Check that the model name exists on Groq, your "
                "messages are correctly formatted, and max_tokens is reasonable.\n\n"
                f"Original error: {original_exc}"
            ) from original_exc

        # Fallback: wrap as generic client error
        raise GroqClientError(
            f"Groq API call failed: {original_exc}"
        ) from original_exc
