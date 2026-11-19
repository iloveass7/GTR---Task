# Samsung Phone Assistant

A small interview project that collects Samsung phone specifications from GSMArena, stores them in PostgreSQL, answers dataset-grounded questions with Gemini, generates reviews with two LangChain agents, and exposes both features through FastAPI.

## What it does

- Scrapes selected Samsung phone detail pages with `requests` and BeautifulSoup.
- Stores phone names, specifications, and listed prices in PostgreSQL.
- Answers natural-language questions from the saved dataset using Gemini.
- Uses a LangChain data agent to retrieve a phone record and a separate review agent to write a review from it.
- Provides an HTTP API for chat, reviews, phone listing, and a health check.

## Project layout

```text
.
├── scraper.py       # GSMArena scraper
├── schema.sql       # PostgreSQL tables
├── chatbot.py       # Dataset-grounded Gemini CLI chatbot
├── review.py        # LangChain retrieval and review agents
├── main.py          # FastAPI application
├── requirements.txt
├── .env.example     # Environment variable template
└── .gitignore
```

## Quick start (Windows PowerShell)

### 1. Install Python packages

From the project directory, activate your Python environment if you use one, then install the dependencies:

```powershell
conda activate dev
python -m pip install -r requirements.txt
```

### 2. Configure PostgreSQL and Gemini

Copy `.env.example` to `.env`, then edit `.env` with your PostgreSQL connection details and Gemini API key:

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
notepad .env
```

Example values:

```dotenv
DATABASE_URL=postgresql://postgres:your_password@localhost:5432/GTR
GEMINI_API_KEY=your_google_ai_studio_key
GEMINI_MODEL=gemini-3.5-flash-lite
```

Create a Gemini API key in [Google AI Studio](https://aistudio.google.com/app/apikey). Keep `.env` private; it is excluded by `.gitignore` and should not be committed or shared.

### 3. Create the database tables

In pgAdmin, select your project database (for example `GTR`), open **Query Tool**, open `schema.sql`, and execute it. The script creates the `phones`, `specifications`, and `prices` tables. It also removes the obsolete `network_technology` column from an older `specifications` table if present.

### 4. Scrape phones

The scraper takes explicit GSMArena phone detail page URLs; it does not crawl the site on its own. For example:

```powershell
python .\scraper.py https://www.gsmarena.com/samsung_galaxy_s23-12082.php https://www.gsmarena.com/samsung_galaxy_s22-11253.php
```

Add up to 15 unique phone URLs per run. Re-running a URL updates its phone and specification records. Run the scraper before using the chatbot, review tool, or API.

## Run the applications

### Dataset-grounded chatbot (Stage 2)

```powershell
python .\chatbot.py
```

Ask questions in natural language, including comparisons and follow-ups. Type `exit` to quit. Gemini is instructed to use only the PostgreSQL records and say when a requested value is missing. The app sends the dataset to Gemini as context; it does not use vector search or web search.

### LangChain review agents (Stage 3)

```powershell
python .\review.py
```

Enter a phone model, such as `Samsung Galaxy S24 Ultra`. The data agent calls a read-only PostgreSQL lookup tool; the review agent writes from the returned record. If the name matches multiple phones, specify a more precise model name. Type `exit` to quit.

### FastAPI (Stage 4)

```powershell
python -m uvicorn main:app --reload
```

Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) to explore and call the endpoints in the interactive API documentation.

## API endpoints

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Check that the API process is responding. |
| `GET` | `/phones` | Return phone names and the count from PostgreSQL. |
| `POST` | `/chat` | Ask a question using the dataset-grounded chatbot. |
| `POST` | `/reviews` | Generate a review for a phone in the dataset. |

### Chat request

Send `POST http://127.0.0.1:8000/chat` with `Content-Type: application/json`:

```json
{
  "message": "Which phone has the largest battery?",
  "history": []
}
```

`history` is optional and accepts up to four prior user/assistant exchanges for follow-up questions. For example:

```json
{
  "message": "How does its camera compare to the S23 Ultra?",
  "history": [
    {
      "user": "Tell me about the Samsung Galaxy S24 Ultra",
      "assistant": "The dataset lists its display, camera, battery, and chipset details."
    }
  ]
}
```

The response is JSON:

```json
{
  "answer": "..."
}
```

### Review request

Send `POST http://127.0.0.1:8000/reviews` with `Content-Type: application/json`:

```json
{
  "phone_name": "Samsung Galaxy S24 Ultra"
}
```

The endpoint returns the review as plain text (`text/plain`), so the review's line breaks are displayed normally instead of escaped inside a JSON string.

## Database structure

- `phones`: one row per phone, including its name and GSMArena source URL.
- `specifications`: one-to-one with `phones`; common fields are columns, and `raw_specs` (JSONB) retains the full labeled specification data.
- `prices`: separate source price observations, including the original text, a parsed currency marker, and a numeric value when parseable.

## Scraper guardrails and data notes

- Only HTTPS phone detail URLs on `www.gsmarena.com` are accepted.
- Pages must identify as Samsung phones and contain expected specification rows; tablet and watch pages are rejected.
- Each run accepts at most 15 unique URLs, uses request timeouts and a 5 MB response limit, and waits two seconds between requests.
- The scraper stores source price text. GSMArena prices may be estimates or list multiple currencies; do not compare numeric values across currencies without normalizing them first.
- Missing specification fields remain empty in PostgreSQL. If GSMArena changes its page structure, the scraper may stop and report that expected data could not be found.
- Camera megapixels, battery capacity, and benchmark results are useful stored facts but do not by themselves prove which phone is objectively “best.”

## Current scope

This is a local interview-project prototype. Scraping is a separate ingestion step, not an API endpoint. The API is intended to run locally for development/demo; it has no user authentication or production deployment configuration.
