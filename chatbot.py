"""Dataset-grounded Samsung phone chatbot using PostgreSQL and Gemini."""

from __future__ import annotations

import json
import os
import re
import sys

import psycopg
from dotenv import load_dotenv
from google import genai

SYSTEM_RULES = """You are a helpful Samsung phone assistant for an interview project.
Answer using ONLY the supplied PostgreSQL dataset. Do not use outside knowledge,
search the web, or invent missing specifications, prices, or conclusions.
The user may ask casually, use abbreviations, ask for comparisons, or refer to a
phone from earlier in the conversation. Interpret their intent flexibly, but
base every factual statement on the supplied records. For comparisons, compare
only fields present for both models. If the dataset lacks a requested value,
say that it is not available in this dataset. If a requested model is absent,
say it is not in this dataset. Do not claim a phone is "best" unless you state
the specific stored field used to make that comparison. For a vague "best
camera" question, use the highest primary main-camera megapixel value as a
limited dataset-based comparison when available, and clearly state that MP
alone does not establish overall photo quality. Be concise and name which
phone each value belongs to."""


def load_phone_data(database_url: str) -> list[dict]:
    query = """
        SELECT p.name, p.source_url,
               s.display_size, s.weight, s.dimensions, s.camera, s.battery,
               s.chipset, s.os, s.storage, s.raw_specs,
               price.price_text, price.observed_at
        FROM phones AS p
        LEFT JOIN specifications AS s ON s.phone_id = p.id
        LEFT JOIN LATERAL (
            SELECT price_text, observed_at
            FROM prices
            WHERE phone_id = p.id
            ORDER BY observed_at DESC
            LIMIT 1
        ) AS price ON TRUE
        ORDER BY p.name
    """
    with psycopg.connect(database_url) as conn, conn.cursor() as cur:
        cur.execute(query)
        columns = [column.name for column in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]


def make_prompt(question: str, phones: list[dict], history: list[tuple[str, str]]) -> str:
    # Datetimes and any Decimal values are rendered as readable strings.
    dataset = json.dumps(phones, default=str, ensure_ascii=False)
    camera_scores = []
    for phone in phones:
        camera = phone.get("camera") or ""
        match = re.search(r"Main Camera:\s*(\d+(?:\.\d+)?)\s*MP", camera, re.I)
        if match:
            camera_scores.append((float(match.group(1)), phone["name"]))
    camera_hint = ""
    if camera_scores:
        highest = max(score for score, _ in camera_scores)
        winners = [name for score, name in camera_scores if score == highest]
        camera_hint = (
            f"\nDATASET COMPARISON NOTE (derived from stored camera strings): "
            f"Highest primary main-camera resolution recorded is {highest:g} MP for "
            f"{', '.join(winners)}. This is only an MP comparison, not a photo-quality score.\n"
        )
    previous = "\n".join(
        f"{role.upper()}: {message}" for pair in history for role, message in zip(("user", "assistant"), pair)
    )
    return (
        f"{SYSTEM_RULES}\n\nDATASET (authoritative source):\n{dataset}\n\n"
        + camera_hint
        + (f"RECENT CONVERSATION:\n{previous}\n\n" if previous else "")
        + f"USER QUESTION:\n{question}\n\nANSWER:"
    )


def main() -> int:
    load_dotenv()
    database_url = os.getenv("DATABASE_URL")
    api_key = os.getenv("GEMINI_API_KEY")
    if not database_url:
        print("DATABASE_URL is missing. Set it in your .env file.", file=sys.stderr)
        return 1
    if not api_key:
        print("GEMINI_API_KEY is missing. Add your AI Studio key to .env.", file=sys.stderr)
        return 1

    model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    try:
        phones = load_phone_data(database_url)
    except psycopg.Error as exc:
        print(f"Could not read phone data from PostgreSQL: {exc}", file=sys.stderr)
        return 1
    if not phones:
        print("No phones found. Run the scraper before starting the chatbot.", file=sys.stderr)
        return 1

    client = genai.Client(api_key=api_key)
    history: list[tuple[str, str]] = []
    print(f"Samsung phone chatbot ready ({len(phones)} phones loaded). Ask a question, or type 'exit'.")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        try:
            response = client.models.generate_content(
                model=model,
                contents=make_prompt(question, phones, history[-4:]),
            )
            answer = (response.text or "I couldn't produce an answer from the available dataset.").strip()
            print(f"Bot: {answer}\n")
            history.append((question, answer))
        except Exception as exc:
            print(f"Gemini request failed: {exc}\n", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
