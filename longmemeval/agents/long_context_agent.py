import json
import os
import sys
from datetime import datetime
from typing import Any, Literal

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI
from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

    model_config = ConfigDict(extra="ignore")


class InputCase(BaseModel):
    case_id: str
    input: str = Field(validation_alias=AliasChoices("input", "question", "query"))
    history: list[Message] = Field(default_factory=list)
    choices: dict[str, str] = Field(default_factory=dict)
    haystack_sessions: list[Any] | None = None
    haystack_dates: list[Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="ignore")


def _parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    for fmt in ("%Y/%m/%d (%a) %H:%M", "%Y/%m/%d %H:%M"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def _flatten_history(case: InputCase) -> list[Message]:
    if case.history:
        return case.history

    sessions = case.haystack_sessions
    dates = case.haystack_dates
    if not sessions and case.metadata:
        sessions = case.metadata.get("haystack_sessions")
        dates = case.metadata.get("haystack_dates")

    if not isinstance(sessions, list):
        return []
    if not isinstance(dates, list):
        dates = []

    ordered_sessions: list[list[Any]] = []
    if dates and len(dates) == len(sessions):
        parsed = []
        parsed_all = True
        for idx, (date, session) in enumerate(zip(dates, sessions)):
            date_str = str(date) if date is not None else None
            parsed_date = _parse_date(date_str)
            if parsed_date is None:
                parsed_all = False
            parsed.append((parsed_date, idx, session))
        if parsed_all:
            parsed.sort(key=lambda item: (item[0], item[1]))
        ordered_sessions = [session for _, _, session in parsed]
    else:
        ordered_sessions = sessions

    history: list[Message] = []
    for session in ordered_sessions:
        if not isinstance(session, list):
            continue
        for msg in session:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role", "user")).strip().lower()
            content = msg.get("content")
            if content is None:
                continue
            if role not in {"system", "user", "assistant"}:
                role = "user"
            history.append(Message(role=role, content=str(content)))
    return history


def build_messages(case: InputCase) -> list[dict[str, str]]:
    system_prompt = (
        "You are a helpful assistant with long-term memory. "
        "Use the conversation history to answer the question accurately."
    )
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]

    for msg in _flatten_history(case):
        messages.append({"role": msg.role, "content": msg.content})

    user_prompt = f"Question: {case.input}"
    if case.choices:
        choices_text = "\n".join(
            f"{key}. {value}" for key, value in sorted(case.choices.items())
        )
        user_prompt += (
            f"\n\nChoices:\n{choices_text}"
            "\n\nRespond with only the choice letter (A, B, C, or D)."
        )

    messages.append({"role": "user", "content": user_prompt})
    return messages


def main() -> None:
    if os.environ.get("OPENAI_API_KEY", "") == "":
        os.environ.pop("OPENAI_API_KEY", None)

    load_dotenv(find_dotenv(usecwd=False))

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    client = OpenAI(api_key=api_key, base_url=base_url)

    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            data = json.loads(line)
            case = InputCase.model_validate(data)
        except Exception:
            continue

        try:
            response = client.chat.completions.create(
                model=model,
                messages=build_messages(case),
            )
            output = response.choices[0].message.content
            print(json.dumps({"output": output, "error": None}))
        except Exception as exc:
            print(json.dumps({"output": None, "error": str(exc)}))

        try:
            sys.stdout.flush()
        except BrokenPipeError:
            return


if __name__ == "__main__":
    main()
