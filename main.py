"""FastAPI endpoints for the Samsung phone chatbot and review agents."""

from __future__ import annotations

import os
from functools import lru_cache

import psycopg
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from google import genai
from pydantic import BaseModel, Field

from chatbot import load_phone_data, make_prompt, retrieve_phone_data
from review import create_agents, make_review

load_dotenv()

app = FastAPI(
    title="Samsung Phone Assistant API",
    description="Chat about phones stored in PostgreSQL or generate a data-grounded review.",
    version="1.0.0",
)


class ChatTurn(BaseModel):
    user: str = Field(min_length=1, max_length=2000)
    assistant: str = Field(min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    history: list[ChatTurn] = Field(default_factory=list, max_length=4)


class ChatResponse(BaseModel):
    answer: str


class ReviewRequest(BaseModel):
    phone_name: str = Field(min_length=1, max_length=200)


class PhoneListResponse(BaseModel):
    phones: list[str]
    count: int


def require_database_url() -> str:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise HTTPException(status_code=503, detail="DATABASE_URL is not configured")
    return database_url


def current_model() -> str:
    return os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")


@lru_cache(maxsize=1)
def get_gemini_client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)


@lru_cache(maxsize=2)
def get_review_agents(model_name: str):
    return create_agents(model_name)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/phones", response_model=PhoneListResponse)
def list_phones() -> PhoneListResponse:
    try:
        phones = load_phone_data(require_database_url())
    except psycopg.Error as exc:
        raise HTTPException(status_code=503, detail="Could not read phones from PostgreSQL") from exc
    names = [phone["name"] for phone in phones]
    return PhoneListResponse(phones=names, count=len(names))


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="Gemini API key is not configured")
    try:
        database_url = require_database_url()
        retrieval_question = request.message
        if request.history:
            retrieval_question = f"{request.history[-1].user} {request.message}"
        phones = retrieve_phone_data(database_url, retrieval_question)
    except psycopg.Error as exc:
        raise HTTPException(status_code=503, detail="Could not read phone data from PostgreSQL") from exc
    if not phones:
        raise HTTPException(status_code=503, detail="No phones are stored in the dataset")

    history = [(turn.user, turn.assistant) for turn in request.history]
    try:
        response = get_gemini_client(api_key).models.generate_content(
            model=current_model(),
            contents=make_prompt(request.message, phones, history),
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Gemini could not complete the request") from exc
    answer = (response.text or "I couldn't produce an answer from the available dataset.").strip()
    return ChatResponse(answer=answer)


@app.post("/reviews", response_class=PlainTextResponse)
def create_phone_review(request: ReviewRequest) -> PlainTextResponse:
    require_database_url()
    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        raise HTTPException(status_code=503, detail="Gemini API key is not configured")
    try:
        data_agent, review_agent = get_review_agents(current_model())
        review_text = make_review(request.phone_name, data_agent, review_agent)
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Review agents could not complete the request") from exc
    return PlainTextResponse(content=review_text)
