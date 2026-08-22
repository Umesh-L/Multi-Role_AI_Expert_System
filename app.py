"""
Multi-Role AI Expert System — Main Streamlit Application
=========================================================

This is the entry point for the Streamlit web interface. Run it with:

    streamlit run app.py

Features:
  1. Role selector sidebar — choose from 8 expert personas
  2. Persistent chat UI (message history kept in st.session_state)
  3. Streaming responses (tokens appear as they arrive from Groq)
  4. Advanced settings panel (model, temperature, max_tokens)
  5. Token / timing usage stats displayed after each response
  6. Role-specific welcome messages and sidebar info cards

Architecture
------------
The app is intentionally split into separate modules for clarity:
  • app.py        → Streamlit UI + session state management (THIS FILE)
  • roles.py      → Role definitions and system prompts
  • groq_client.py→ Groq API wrapper with typed error classes
  • requirements.txt → Dependencies for pip install
  • .streamlit/secrets.toml → GROQ_API_KEY (local dev)
  • .streamlit/config.toml → Streamlit display configuration
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import streamlit as st

# -----------------------------------------------------------------------------
# Internal module imports
# -----------------------------------------------------------------------------
from roles import (
    ROLES,
    get_role_metadata,
    get_role_names,
    get_system_prompt,
)
from groq_client import (
    DEFAULT_MODEL,
    GroqClient,
    GroqAuthenticationError,
    GroqClientError,
    GroqRateLimitError,
    GroqServiceError,
    GroqValidationError,
    build_messages,
)

# -----------------------------------------------------------------------------
# Page config — must be first Streamlit call
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Multi-Role AI Expert System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://console.groq.com/docs",
        "Report a bug": "https://github.com/",
        "About": (
            "Multi-Role AI Expert System powered by Groq API.\n"
            "Choose from 8 expert personas and get instant, production-grade advice."
        ),
    },
)


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================
# Streamlit reruns the script on every user interaction.
# We use st.session_state to preserve data between reruns.
# =============================================================================

def _init_session_state() -> None:
    """Create missing session-state keys with sensible defaults."""

    # --- Core conversation state ---
    if "messages" not in st.session_state:
        # Each message: {"role": "user"|"assistant", "content": str,
        #                "usage": {...optional...}, "timestamp": float}
        st.session_state.messages = []

    if "current_role_key" not in st.session_state:
        st.session_state.current_role_key = "software_engineer"

    if "conversation_started" not in st.session_state:
        st.session_state.conversation_started = False

    # --- Settings ---
    # CRITICAL: ALWAYS force the model to DEFAULT_MODEL.
    # Prevents stale "llama-3.1-70b-versatile" (decommissioned) from persisting
    # in user sessions across code updates — no user-facing model choice anymore.
    st.session_state.model = DEFAULT_MODEL

    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.7

    if "max_tokens" not in st.session_state:
        st.session_state.max_tokens = 4096

    if "stream_mode" not in st.session_state:
        st.session_state.stream_mode = True

    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None


_init_session_state()


# =============================================================================
# GROQ CLIENT FACTORY — MODULE-LEVEL SINGLETON (no @st.cache_resource!)
# =============================================================================
#
# ═══════════════════════════════════════════════════════════════════════════════
# ARCHITECTURAL RATIONALE: Why we deliberately AVOID @st.cache_resource here
#
#   BUG THAT WAS HAPPENING ON STREAMLIT CLOUD:
#     1. User pushes app to GitHub → connects Streamlit Cloud → app boots
#     2. Streamlit Cloud serves the first HTTP request BEFORE the user has
#        pasted their GROQ_API_KEY into the Settings → Secrets web UI
#     3. Old code: @st.cache_resource caught GroqAuthenticationError, returned
#        None, and CACHED THAT None FOREVER for the Python process lifetime
#     4. User THEN goes to Settings → Secrets, pastes the key correctly, saves,
#        and reruns the app → get_groq_client() still returns the CACHED None
#        → "GROQ_API_KEY missing" banner persists FOREVER until a full reboot
#        of the app from the Streamlit dashboard.
#
#   ADDITIONAL PROBLEM: @st.cache_resource also CACHES EXCEPTIONS by default in
#   newer Streamlit releases — so even if we let GroqAuthenticationError bubble
#   up, it would replay on every rerun until reboot.
#
#   SOLUTION:
#     • Keep module-level `_CLIENT_INSTANCE: Optional[GroqClient] = None`
#     • On SUCCESSFUL GroqClient() construction → write to singleton, reuse
#     • On key-missing → return None WITHOUT writing singleton → next rerun
#       will RE-ATTEMPT key resolution from scratch. This means the instant
#       the user saves their secret in Streamlit Cloud + clicks Rerun, the
#       client is built and stored, no reboot required.
#
#   PERFORMANCE JUSTIFICATION: GroqClient.__init__ is TRIVIALLY FAST. The
#   heavy httpx SDK client is constructed LAZILY inside Groq SDK on the FIRST
#   actual chat.completions.create() call — not in __init__. So we pay the
#   cost of key-resolution (dict lookups) only on reruns that happen BEFORE
#   the user enters their key (usually 0–2 reruns total).
# ═══════════════════════════════════════════════════════════════════════════════

_CLIENT_INSTANCE: Optional[GroqClient] = None


def get_groq_client() -> Optional[GroqClient]:
    """
    Return a shared GroqClient, or None if GROQ_API_KEY is not configured yet.

    Behavior:
      * Once a client is successfully built, it is stored in a module singleton
        and reused for every subsequent rerun.
      * If the key is missing on a given run, **no state is persisted** — every
        rerun re-attempts key resolution so newly-entered Streamlit Cloud
        secrets are picked up immediately on next Rerun.
    """
    global _CLIENT_INSTANCE

    # Fast path: client is already built — reuse forever
    if _CLIENT_INSTANCE is not None:
        return _CLIENT_INSTANCE

    # Slow path: try to construct. Retried on every rerun until success.
    try:
        _CLIENT_INSTANCE = GroqClient()
        return _CLIENT_INSTANCE
    except GroqAuthenticationError:
        # Intentionally do NOT assign to _CLIENT_INSTANCE here.
        # We want the NEXT rerun to try key-lookup again from scratch.
        return None


# =============================================================================
# SIDEBAR: ROLE SELECTOR + SETTINGS
# =============================================================================

def render_sidebar() -> None:
    """Render the left sidebar: title, role picker, settings."""

    with st.sidebar:
        # --- Header -------------------------------------------------------
        st.header("🧠 Multi-Role AI Expert")
        st.caption(
            "Powered by Groq · Ultra-fast LLM inference"
        )
        st.divider()

        # --- Role Selector ------------------------------------------------
        st.subheader("🎭 Choose an Expert")

        role_options = get_role_names()  # list of (key, name, icon)
        display_labels = [f"{icon}  {name}" for _, name, icon in role_options]
        keys_only = [key for key, _, _ in role_options]

        # Find the index of the currently selected role
        current_idx = keys_only.index(st.session_state.current_role_key) \
            if st.session_state.current_role_key in keys_only else 0

        selected_label = st.selectbox(
            "Select expert role:",
            options=display_labels,
            index=current_idx,
            help="Pick the persona the AI will adopt for this conversation.",
        )

        # Map the selected label back to the role key
        new_role_key = keys_only[display_labels.index(selected_label)]

        # If role changed, reset conversation (the system prompt is changing)
        if new_role_key != st.session_state.current_role_key:
            st.session_state.current_role_key = new_role_key
            st.session_state.messages = []
            st.session_state.conversation_started = False
            st.rerun()

        # Show current role info card
        role_meta = get_role_metadata(st.session_state.current_role_key)
        with st.container(border=True):
            st.markdown(
                f"### {role_meta['icon']} {role_meta['name']}"
            )
            st.caption(role_meta["description"])

        st.divider()

        # --- Advanced Settings -------------------------------------------
        with st.expander("⚙️ Advanced Settings", expanded=False):
            # Fixed model (no user choice) — display what we're using.
            # See DEFAULT_MODEL in groq_client.py to change it.
            with st.container(border=True):
                st.markdown("🤖 **Groq Model (fixed)**")
                st.code(DEFAULT_MODEL, language="text")
                st.caption(
                    "Model is locked to prevent decommissioning errors. "
                    "To switch it, edit `DEFAULT_MODEL` in `groq_client.py`."
                )

            temp_val = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=2.0,
                value=float(st.session_state.temperature),
                step=0.1,
                help=(
                    "Lower = more deterministic / factual. "
                    "Higher = more creative / varied."
                ),
            )
            st.session_state.temperature = temp_val

            max_tok = st.slider(
                "Max Output Tokens",
                min_value=256,
                max_value=8192,
                value=int(st.session_state.max_tokens),
                step=256,
                help="Maximum number of tokens the AI can generate per reply.",
            )
            st.session_state.max_tokens = max_tok

            st.toggle(
                "Stream responses",
                value=st.session_state.stream_mode,
                key="stream_mode",
                help="Show tokens as they arrive vs wait for full response.",
            )

        st.divider()

        # --- Action Buttons ----------------------------------------------
        col1, col2 = st.columns(2)
        with col1:
            if st.button(
                "🗑️ Clear Chat",
                use_container_width=True,
                help="Reset the current conversation history.",
            ):
                st.session_state.messages = []
                st.session_state.conversation_started = False
                st.rerun()
        with col2:
            if st.button(
                "🔄 New Topic",
                use_container_width=True,
                help="Same role, fresh conversation.",
            ):
                st.session_state.messages = []
                st.session_state.conversation_started = False
                st.rerun()

        st.divider()

        # --- API Key Status -----------------------------------------------
        client = get_groq_client()
        if client is None:
            st.error(
                "⚠️ **GROQ_API_KEY missing**\n\n"
                "Set it in `.streamlit/secrets.toml` locally, or in the "
                "Streamlit Community Cloud dashboard under **Settings → Secrets** "
                "after deploying to GitHub.\n\n"
                "Get a key at: https://console.groq.com/keys"
            )
        else:
            st.success("✅ Groq API key loaded")

        st.caption("v1.0 · Built with Streamlit + Groq")


# =============================================================================
# WELCOME / EMPTY-STATE HEADER
# =============================================================================

def render_welcome_header() -> None:
    """
    Render the hero banner + suggested starter prompts.
    Only shown when the conversation is empty.
    """
    role_meta = get_role_metadata(st.session_state.current_role_key)
    role_key = st.session_state.current_role_key

    # --- Hero banner ------------------------------------------------------
    st.markdown(
        f"""
        <div style='padding: 1.5rem 2rem; border-radius: 1rem;
                    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
                    color: white; margin-bottom: 2rem;'>
            <h2 style='margin: 0 0 0.5rem 0;'>
                {role_meta['icon']} {role_meta['name']} — at your service
            </h2>
            <p style='margin: 0; opacity: 0.95; font-size: 1.05rem;'>
                {role_meta['description']}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Suggested starter prompts (role-specific) -----------------------
    STARTER_PROMPTS: Dict[str, List[str]] = {
        "software_engineer": [
            "Explain the SOLID principles with Python examples",
            "Design a REST API for a task management system",
            "Review this code for bugs and performance issues",
            "Compare microservices vs monolithic architecture",
        ],
        "data_scientist": [
            "Walk me through building a churn prediction model",
            "How do I detect and treat outliers in a dataset?",
            "Explain the bias-variance tradeoff with a concrete example",
            "Design an A/B testing framework for a new feature",
        ],
        "legal_advisor": [
            "What key clauses should a SaaS MSA include?",
            "Explain the difference between copyright and trademark",
            "What do I need to know about GDPR compliance?",
            "What are the risks of using open source code commercially?",
        ],
        "marketing_strategist": [
            "Create a go-to-market plan for a new mobile app",
            "How do I differentiate from competitors in a crowded market?",
            "Write a content strategy for a B2B SaaS company",
            "How should I allocate a $10k monthly ad budget?",
        ],
        "financial_analyst": [
            "Walk me through building a 3-statement financial model",
            "How do I value an early-stage startup?",
            "Explain WACC and how it's used in DCF analysis",
            "What KPIs should a SaaS CEO track monthly?",
        ],
        "health_coach": [
            "Design a realistic 12-week strength training routine for a beginner",
            "Give me a 7-day healthy meal plan for busy professionals",
            "How can I improve my sleep quality with science-backed habits?",
            "What's the best way to lose fat while preserving muscle?",
        ],
        "career_coach": [
            "Rewrite my resume bullet points using the STAR method",
            "How do I negotiate a higher salary after a job offer?",
            "Help me transition from coding to product management",
            "What are the best strategies to get promoted in 2026?",
        ],
        "product_manager": [
            "Write a PRD for a new AI-powered feature",
            "How do I prioritize my backlog when everything feels urgent?",
            "Walk me through a complete product discovery process",
            "How do I know my product has achieved product-market fit?",
        ],
    }

    st.markdown("#### 💡 Try asking:")
    prompts = STARTER_PROMPTS.get(role_key, STARTER_PROMPTS["software_engineer"])

    # Render prompts in a 2x2 grid of buttons
    cols = st.columns(2)
    for idx, prompt_text in enumerate(prompts):
        with cols[idx % 2]:
            if st.button(
                prompt_text,
                key=f"starter_{role_key}_{idx}",
                use_container_width=True,
            ):
                st.session_state.pending_prompt = prompt_text
                st.rerun()


# =============================================================================
# MESSAGE HISTORY RENDERING
# =============================================================================

def render_messages() -> None:
    """Draw all prior messages (user + assistant) in the main chat area."""
    for msg in st.session_state.messages:
        role = msg["role"]
        avatar = "👤" if role == "user" else get_role_metadata(st.session_state.current_role_key)["icon"]

        with st.chat_message(role, avatar=avatar):
            st.markdown(msg["content"])

            # Show token/time stats if available (assistant messages only)
            if role == "assistant" and "usage" in msg and msg["usage"]:
                u = msg["usage"]
                with st.expander("📊 Response stats", expanded=False):
                    c1, c2, c3 = st.columns(3)
                    c1.metric(
                        "Tokens",
                        f"{u.get('prompt_tokens', 0)}→{u.get('completion_tokens', 0)}",
                        help="Prompt tokens → Completion tokens",
                    )
                    c2.metric(
                        "Latency",
                        f"{u.get('total_time', 0):.2f}s",
                        help="Total round-trip time (seconds)",
                    )
                    c3.metric(
                        "Speed",
                        (
                            f"{u.get('completion_tokens', 0) / u.get('completion_time', 1):.0f} t/s"
                            if u.get("completion_time", 0) > 0
                            else "—"
                        ),
                        help="Tokens generated per second (completion only)",
                    )


# =============================================================================
# CORE CHAT LOGIC
# =============================================================================

def _conversation_history_for_groq() -> List[Dict[str, str]]:
    """Return user/assistant messages (no system, no usage) for Groq API."""
    return [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
        if m["role"] in ("user", "assistant")
    ]


def run_user_prompt(user_text: str) -> None:
    """
    Execute a full user→assistant turn. This is called both by the
    chat_input widget AND by the starter prompt buttons.
    """
    if not user_text or not user_text.strip():
        return

    # 1. Append user message to history
    st.session_state.messages.append({
        "role": "user",
        "content": user_text.strip(),
        "timestamp": time.time(),
    })
    st.session_state.conversation_started = True

    # 2. Render user message immediately (before rerun)
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_text.strip())

    # 3. Generate & render assistant response
    role_key = st.session_state.current_role_key
    system_prompt = get_system_prompt(role_key)

    messages_for_api = build_messages(
        system_prompt=system_prompt,
        user_query=user_text.strip(),
        conversation_history=_conversation_history_for_groq(),
    )

    client = get_groq_client()
    avatar = get_role_metadata(role_key)["icon"]

    with st.chat_message("assistant", avatar=avatar):
        response_placeholder = st.empty()

        if client is None:
            # No API key available — show the error in-place
            st.error(
                "🚫 **Groq API key not configured.**\n\n"
                "Add `GROQ_API_KEY = \"gsk_yourkey\"` to `.streamlit/secrets.toml` "
                "or set it in Streamlit Cloud dashboard → Secrets.\n\n"
                "Get your free key: https://console.groq.com/keys"
            )
            return

        full_content = ""
        usage_capture = {}

        try:
            if st.session_state.stream_mode:
                # --- STREAMING PATH ---------------------------------------
                response_placeholder.markdown("⏳ Thinking...")
                chunk_buffer: List[str] = []

                stream = client.chat_stream(
                    messages=messages_for_api,
                    model=DEFAULT_MODEL,
                    temperature=st.session_state.temperature,
                    max_tokens=st.session_state.max_tokens,
                )
                for chunk in stream:
                    chunk_buffer.append(chunk)
                    # Update the UI periodically (every chunk is fine; Streamlit throttles)
                    response_placeholder.markdown("".join(chunk_buffer))
                full_content = "".join(chunk_buffer)

            else:
                # --- NON-STREAMING PATH -----------------------------------
                with st.spinner("⏳ Generating response..."):
                    response = client.chat(
                        messages=messages_for_api,
                        model=DEFAULT_MODEL,
                        temperature=st.session_state.temperature,
                        max_tokens=st.session_state.max_tokens,
                    )
                full_content = response.content
                usage_capture = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                    "prompt_time": response.usage.prompt_time,
                    "completion_time": response.usage.completion_time,
                    "total_time": response.usage.total_time,
                }
                response_placeholder.markdown(full_content)

        except GroqAuthenticationError as exc:
            st.error(f"🔐 **Authentication Error:**\n\n{exc}")
            return
        except GroqRateLimitError as exc:
            st.warning(f"⏳ **Rate Limited:**\n\n{exc}")
            return
        except GroqServiceError as exc:
            st.error(f"⚠️ **Service Unavailable:**\n\n{exc}")
            return
        except GroqValidationError as exc:
            st.error(f"❌ **Invalid Request:**\n\n{exc}")
            return
        except GroqClientError as exc:
            st.error(f"🔥 **API Error:**\n\n{exc}")
            return

        # Safety: if model hallucinated an empty string, show placeholder
        if not full_content.strip():
            full_content = "_(No response received.)_"

        # Final UI pass (streaming: ensure markdown is fully rendered)
        response_placeholder.markdown(full_content)

        # For streaming mode, fetch a quick non-stream completion just for
        # usage stats? Too wasteful — instead we fall back to estimating
        # token counts locally if we really want stats. For now, show a
        # simplified usage card when we have the data.
        usage_entry = usage_capture if usage_capture else None

        # 4. Append assistant response to session history
        new_assistant_msg: Dict = {
            "role": "assistant",
            "content": full_content,
            "timestamp": time.time(),
        }
        if usage_entry:
            new_assistant_msg["usage"] = usage_entry
            # Also render the stats expander below the response for non-stream
            with st.expander("📊 Response stats", expanded=False):
                u = usage_entry
                c1, c2, c3 = st.columns(3)
                c1.metric("Tokens", f"{u['prompt_tokens']}→{u['completion_tokens']}")
                c2.metric("Latency", f"{u['total_time']:.2f}s")
                speed = (
                    f"{u['completion_tokens'] / u['completion_time']:.0f} t/s"
                    if u["completion_time"] > 0 else "—"
                )
                c3.metric("Speed", speed)

        st.session_state.messages.append(new_assistant_msg)


def handle_chat_input() -> None:
    """Bind the st.chat_input widget to the run_user_prompt handler."""
    prompt = st.chat_input(
        placeholder=(
            f"Ask the {get_role_metadata(st.session_state.current_role_key)['name']} anything..."
        ),
    )
    if prompt:
        run_user_prompt(prompt)


# =============================================================================
# MAIN LAYOUT
# =============================================================================

def main() -> None:
    """Top-level page assembly."""
    render_sidebar()

    # --- Pending prompt dispatcher (runs OUTSIDE any column/layout context) ---
    # This fixes the bug where clicking a starter button on the right half
    # would constrain the response to only that 50%-width column. We store
    # the prompt in session_state + rerun, then consume it here at full-width.
    if st.session_state.pending_prompt is not None:
        prompt_text = st.session_state.pending_prompt
        st.session_state.pending_prompt = None
        run_user_prompt(prompt_text)

    # --- Main content area -----------------------------------------------
    st.title("🤝 Consult an Expert AI")
    st.caption(
        "Pick a role on the left, then ask your question. "
        "Each expert has specialized knowledge and tailored responses."
    )
    st.markdown("---")

    # Show welcome banner OR render prior messages (mutually exclusive)
    if not st.session_state.conversation_started and len(st.session_state.messages) == 0:
        render_welcome_header()
    else:
        render_messages()

    # The chat input is always rendered at the bottom
    handle_chat_input()


if __name__ == "__main__":
    main()
