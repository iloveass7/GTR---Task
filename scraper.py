"""Small, allowlist-driven GSMArena Samsung phone scraper."""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from urllib.parse import urlparse

import psycopg
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

BASE_HOST = "www.gsmarena.com"
MAX_PHONES = 15
USER_AGENT = "SamsungPhoneResearchBot/1.0"
ALLOWED_FIELDS = {
    "Dimensions": "dimensions",
    "Weight": "weight",
    "Size": "display_size",
    "OS": "os",
    "Chipset": "chipset",
    "Internal": "storage",
    "Camera": "camera",
    "Battery": "battery",
}


@dataclass
class Phone:
    name: str
    url: str
    specs: dict[str, str]
    raw_specs: dict[str, str]
    price_text: str | None


def validate_url(url: str) -> str:
    """Only accept direct GSMArena phone detail pages, never arbitrary URLs."""
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != BASE_HOST:
        raise ValueError(f"Only https://{BASE_HOST}/ phone pages are allowed: {url}")
    # GSMArena encodes plus signs in Plus model names as %2B in the URL.
    if not re.fullmatch(r"/[a-z0-9_]+(?:%2b|\+)?-\d+\.php", parsed.path, re.I):
        raise ValueError(f"Expected a GSMArena phone detail URL: {url}")
    if parsed.query or parsed.fragment:
        raise ValueError(f"URL query strings and fragments are not allowed: {url}")
    return f"https://{BASE_HOST}{parsed.path}"


def fetch_phone(url: str, session: requests.Session) -> Phone:
    url = validate_url(url)
    response = session.get(url, timeout=(5, 20))
    response.raise_for_status()
    if len(response.content) > 5_000_000:
        raise ValueError("Response was unexpectedly large; refusing to parse it")

    soup = BeautifulSoup(response.text, "html.parser")
    title = soup.select_one("h1")
    name = title.get_text(" ", strip=True) if title else ""
    if not name or not name.lower().startswith("samsung "):
        raise ValueError(f"Page title is not a Samsung phone: {url}")
    if re.search(r"\b(tablet|watch|tv)\b", name, re.I):
        raise ValueError(f"Page appears to be a non-phone Samsung product: {name}")

    raw: dict[str, str] = {}
    section = ""
    camera_parts: list[str] = []
    for row in soup.select("table tr"):
        heading = row.find("th")
        if heading:
            section = heading.get_text(" ", strip=True)
            continue
        label = row.select_one("td.ttl")
        value = row.select_one("td.nfo")
        if not label or not value:
            continue
        key = label.get_text(" ", strip=True)
        text = " ".join(value.stripped_strings)
        if key and text:
            # Keep section context so labels such as "Single" and "Triple"
            # can be interpreted as camera configuration rows.
            raw_key = f"{section} - {key}" if section else key
            raw[raw_key] = text
            raw[key] = text
            if section.lower() in {"main camera", "selfie camera"} and key.lower() in {
                "single", "dual", "triple", "quad"
            }:
                camera_parts.append(f"{section}: {text}")

    specs: dict[str, str] = {}
    for label, column in ALLOWED_FIELDS.items():
        if label in raw:
            specs[column] = raw[label]
    if camera_parts:
        specs["camera"] = " | ".join(camera_parts)
    if not specs:
        raise ValueError(f"No expected specification rows found; page layout may have changed: {url}")

    # GSMArena's Misc > Price is often an approximate launch price, not a live retail quote.
    price_text = raw.get("Price")
    return Phone(name=name, url=url, specs=specs, raw_specs=raw, price_text=price_text)


def parse_price(price_text: str) -> tuple[str | None, Decimal | None]:
    currency_match = re.search(r"(\$|€|£|₹|৳|\bUSD\b|\bEUR\b|\bGBP\b)", price_text)
    currency = currency_match.group(1) if currency_match else None
    number_match = re.search(r"[\d,]+(?:\.\d+)?", price_text)
    if not number_match:
        return currency, None
    try:
        return currency, Decimal(number_match.group().replace(",", ""))
    except InvalidOperation:
        return currency, None


def save_phone(conn: psycopg.Connection, phone: Phone) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO phones (name, source_url) VALUES (%s, %s)
               ON CONFLICT (source_url) DO UPDATE SET name = EXCLUDED.name,
                 scraped_at = NOW() RETURNING id""",
            (phone.name, phone.url),
        )
        phone_id = cur.fetchone()[0]
        cur.execute(
            """INSERT INTO specifications
               (phone_id, display_size, weight, dimensions, camera, battery, chipset,
                os, storage, raw_specs)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
               ON CONFLICT (phone_id) DO UPDATE SET
                 display_size=EXCLUDED.display_size, weight=EXCLUDED.weight,
                 dimensions=EXCLUDED.dimensions, camera=EXCLUDED.camera,
                 battery=EXCLUDED.battery, chipset=EXCLUDED.chipset,
                 os=EXCLUDED.os, storage=EXCLUDED.storage,
                 raw_specs=EXCLUDED.raw_specs""",
            (phone_id, phone.specs.get("display_size"), phone.specs.get("weight"),
             phone.specs.get("dimensions"), phone.specs.get("camera"),
             phone.specs.get("battery"), phone.specs.get("chipset"),
             phone.specs.get("os"), phone.specs.get("storage"),
             psycopg.types.json.Jsonb(phone.raw_specs)),
        )
        if phone.price_text:
            currency, amount = parse_price(phone.price_text)
            cur.execute(
                """INSERT INTO prices (phone_id, price_text, currency, amount)
                   VALUES (%s, %s, %s, %s) ON CONFLICT (phone_id, price_text) DO NOTHING""",
                (phone_id, phone.price_text, currency, amount),
            )


def main() -> int:
    # Load a project .env when present; existing shell environment variables win.
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("urls", nargs="+", help="Explicit GSMArena Samsung phone detail URLs")
    args = parser.parse_args()
    if len(args.urls) > MAX_PHONES:
        parser.error(f"Maximum {MAX_PHONES} phone URLs per run")
    try:
        urls = list(dict.fromkeys(validate_url(url) for url in args.urls))
    except ValueError as exc:
        parser.error(str(exc))
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        parser.error("Set DATABASE_URL (see .env.example)")

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "text/html"})
    try:
        with psycopg.connect(database_url) as conn:
            for index, url in enumerate(urls):
                phone = fetch_phone(url, session)
                save_phone(conn, phone)
                conn.commit()
                print(f"Saved {phone.name}" + (f" | price: {phone.price_text}" if phone.price_text else " | no listed price"))
                if index < len(urls) - 1:
                    time.sleep(2)  # Keep requests polite.
    except (requests.RequestException, psycopg.Error, ValueError) as exc:
        print(f"Scrape stopped: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
