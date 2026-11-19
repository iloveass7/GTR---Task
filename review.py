"""Stage 3: LangChain data-retrieval and product-review agents."""

from __future__ import annotations

import json
import os
import re
import sys

import psycopg
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI


@tool
def get_phone_specs(phone_name: str) -> str:
    """Look up Samsung phone specs and the latest listed price from PostgreSQL.

    Use this read-only tool with a phone model name such as "Galaxy S24 Ultra".
    It returns matching database records, or a message when no match is found.
    """
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        return "DATABASE_URL is not configured."

    term = phone_name.strip()
    if not term:
        return "Enter a phone model name."
    # Ignore common brand words so "Samsung S27 Ultra" can match the stored
    # name "Samsung Galaxy S27 Ultra". Remaining model tokens are parameterized.
    tokens = [
        token for token in re.findall(r"[a-z0-9]+", term.lower())
        if token not in {"samsung", "galaxy"}
    ]
    if not tokens:
        return "Please include a phone model, such as S24 Ultra."
    token_conditions = " AND ".join("p.name ILIKE %s" for _ in tokens)
    query = f"""
        SELECT p.name, p.source_url,
               s.display_size, s.weight, s.dimensions, s.camera, s.battery,
               s.chipset, s.os, s.storage, s.raw_specs,
               price.price_text
        FROM phones AS p
        LEFT JOIN specifications AS s ON s.phone_id = p.id
        LEFT JOIN LATERAL (
            SELECT price_text
            FROM prices
            WHERE phone_id = p.id
            ORDER BY observed_at DESC
            LIMIT 1
        ) AS price ON TRUE
        WHERE {token_conditions}
        ORDER BY (lower(p.name) = lower(%s)) DESC, p.name
        LIMIT 5
    """
    try:
        with psycopg.connect(database_url) as conn, conn.cursor() as cur:
            cur.execute(query, tuple(f"%{token}%" for token in tokens) + (term,))
            columns = [column.name for column in cur.description]
            records = [dict(zip(columns, row)) for row in cur.fetchall()]
    except psycopg.Error as exc:
        return f"Database lookup failed: {exc.__class__.__name__}"

    if not records:
        return f"No phone matching {term!r} was found in the dataset."
    return json.dumps(records, default=str, ensure_ascii=False)


def content_text(content: object) -> str:
    """Convert LangChain message content to plain text for the next agent."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            item.get("text", "") for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    return str(content)


def clean_review(text: str) -> str:
    """Remove common Markdown markers and literal escaped newlines."""
    text = text.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\r", "\n")
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", text)
    text = text.replace("**", "").replace("__", "").replace("`", "")
    text = re.sub(r"(?m)^\s*[*+-]\s+", "", text)
    text = re.sub(r"(?<!\w)[*_](.*?)[*_](?!\w)", r"\1", text)
    return text.strip()


def create_agents(model_name: str):
    model = ChatGoogleGenerativeAI(model=model_name)
    data_agent = create_agent(
        model=model,
        tools=[get_phone_specs],
        system_prompt=(
            "You are the phone data retrieval agent. For every request, call the "
            "get_phone_specs tool. Do not invent or supplement facts. If the tool "
            "returns multiple matches, ask the user to choose; if it returns no "
            "match, report that clearly."
        ),
    )
    review_agent = create_agent(
        model=model,
        tools=[],
        system_prompt=(
            "You are a Samsung phone review writer. Write a balanced, useful "
            "product review using only the supplied database record. Cover the "
            "display, design/build (when available), performance, cameras, battery, "
            "storage, and listed price where present. Do not add facts from your "
            "general knowledge or claim subjective test results not in the record. "
            "Format the result as plain text with short section labels and paragraphs. "
            "Do not use Markdown headings, bold/italic markers, backticks, bullets, "
            "or literal escape sequences such as \\n. "
            "Use the source wording for OS and update support: do not merge the "
            "launch OS with a later upgrade or current UI version. Do not invent "
            "multiple weight variants or reconcile conflicting source values; "
            "report only an unambiguous value, or note the source ambiguity. "
            "Clearly state when an important detail is missing, and preserve the "
            "price text/currencies as stored."
        ),
    )
    return data_agent, review_agent


def make_review(phone_request: str, data_agent, review_agent) -> str:
    retrieved = data_agent.invoke({"messages": [{"role": "user", "content": phone_request}]})
    tool_result = next(
        (
            content_text(message.content)
            for message in reversed(retrieved["messages"])
            if isinstance(message, ToolMessage) and message.name == "get_phone_specs"
        ),
        None,
    )
    if not tool_result:
        final_message = retrieved["messages"][-1]
        return content_text(final_message.content)
    try:
        records = json.loads(tool_result)
    except json.JSONDecodeError:
        return tool_result
    if not isinstance(records, list) or not records:
        return tool_result
    normalized_request = " ".join(phone_request.casefold().split())
    exact_matches = [
        record for record in records
        if " ".join(record.get("name", "").casefold().split()) == normalized_request
    ]
    if exact_matches:
        # The lookup also returns broader token matches (for example, S25 Ultra
        # when the requested model is S25). Prefer the exact stored model name.
        records = [exact_matches[0]]
        tool_result = json.dumps(records, default=str, ensure_ascii=False)
    if len(records) > 1:
        names = ", ".join(record["name"] for record in records)
        return f"I found multiple matches: {names}. Please specify which phone you mean."

    result = review_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Write a detailed review of this phone using only this database "
                        f"record. The user's requested phone/review: {phone_request}\n\n"
                        f"DATABASE RECORD:\n{tool_result}"
                    ),
                }
            ]
        }
    )
    return clean_review(content_text(result["messages"][-1].content))


def main() -> int:
    load_dotenv()
    if not os.getenv("DATABASE_URL"):
        print("DATABASE_URL is missing. Set it in your .env file.", file=sys.stderr)
        return 1
    if not os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
        print("GEMINI_API_KEY is missing. Add your AI Studio key to .env.", file=sys.stderr)
        return 1

    model_name = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    data_agent, review_agent = create_agents(model_name)
    print("LangChain review agents ready. Type a phone model to review, or 'exit'.")
    while True:
        try:
            request = input("Phone to review: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if request.lower() in {"exit", "quit"}:
            break
        if not request:
            continue
        print("\nRetrieving phone data and writing a review...\n")
        try:
            print(make_review(request, data_agent, review_agent))
        except Exception as exc:
            print(f"Review generation failed: {exc}", file=sys.stderr)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
