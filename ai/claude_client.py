"""
Base Claude API client. All AI features go through this module.
Supports single-turn calls and multi-turn conversation with history.
"""
import json
from typing import Generator
import anthropic

from config.settings import ANTHROPIC_API_KEY, CLAUDE_MODEL, CLAUDE_MAX_TOKENS


def _get_client() -> anthropic.Anthropic:
    if not ANTHROPIC_API_KEY:
        raise ValueError(
            "ANTHROPIC_API_KEY is not set. "
            "Add it to your environment variables or .env file."
        )
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def call_claude(
    system_prompt: str,
    user_message: str,
    temperature: float = 0.3,
    max_tokens: int = CLAUDE_MAX_TOKENS,
) -> str:
    """Single-turn call. Returns response text."""
    client = _get_client()
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )
    return response.content[0].text


def call_claude_with_history(
    system_prompt: str,
    history: list[dict],
    user_message: str,
    temperature: float = 0.5,
    max_tokens: int = CLAUDE_MAX_TOKENS,
) -> str:
    """
    Multi-turn call with conversation history.
    history: list of {"role": "user"|"assistant", "content": "..."}
    """
    client = _get_client()
    messages = list(history) + [{"role": "user", "content": user_message}]
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system_prompt,
        messages=messages,
    )
    return response.content[0].text


def stream_claude(
    system_prompt: str,
    user_message: str,
    history: list[dict] | None = None,
    temperature: float = 0.5,
) -> Generator[str, None, None]:
    """
    Streaming call — yields text chunks as they arrive.
    Use with Streamlit's st.write_stream().
    """
    client = _get_client()
    messages = list(history or []) + [{"role": "user", "content": user_message}]
    with client.messages.stream(
        model=CLAUDE_MODEL,
        max_tokens=CLAUDE_MAX_TOKENS,
        temperature=temperature,
        system=system_prompt,
        messages=messages,
    ) as stream:
        for text in stream.text_stream:
            yield text


def call_claude_json(
    system_prompt: str,
    user_message: str,
    temperature: float = 0.1,
) -> list | dict:
    """
    Calls Claude expecting a JSON response. Parses and returns it.
    The system_prompt must instruct Claude to return only valid JSON.
    """
    raw = call_claude(system_prompt, user_message, temperature=temperature)

    # Strip markdown code fences if present
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:])
        if text.endswith("```"):
            text = text[:-3].strip()

    return json.loads(text)
