
import os
import re
import time
import logging
import urllib.robotparser as robotparser
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

log = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; AcademicContactResearchBot/1.0; "
        "+contact-for-purpose-of-academic-outreach)"
    )
}

REQUEST_DELAY_SECONDS = 2.0
REQUEST_TIMEOUT = 15
MAX_RETRIES = 2

UNIVERSITIES = [
    {
        "name": "University of Toronto",
        "country": "Canada",
        "field": "Computer Science",
        "faculty_urls": [
            "https://web.cs.toronto.edu/people/faculty-directory",
        ],
    },
    {
        "name": "University of British Columbia",
        "country": "Canada",
        "field": "Computer Science",
        "faculty_urls": [
            "https://www.cs.ubc.ca/people/faculty",
        ],
    },

    {
        "name": "RWTH Aachen",
        "country": "Germany",
        "field": "Computer Science",
        "faculty_urls": [
            "https://www.forum.informatik.rwth-aachen.de/staff/"
        ],
    },
    {
        "name": "University of Padova",
        "country": "Italy",
        "field": "Computer Science",
        "faculty_urls": [
              "https://www.dei.unipd.it/en/people"
        ],
    },
    {
        "name": "University of Trento",
        "country": "Italy",
        "field": "Computer Science",
        "faculty_urls": [
            "https://mmlab.science.unitn.it/staff"
        ],
    },
]

IGNORE_EMAIL_DOMAINS_SUBSTRINGS = [
    "example.com",
    "sentry.io",
    "wixpress.com"
]

IGNORE_NAME_WORDS = {
    "home",
    "contact",
    "search",
    "login",
    "menu",
    "faculty",
    "staff",
    "directory",
    "people",
    "about",
    "news",
    "events",
    "admissions",
}

EMAIL_REGEX = re.compile(
    r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
)


TITLE_KEYWORDS_REGEX = re.compile(
    r"\b(Professor|Assistant|Associate|Lecturer|Instructor|Chair|Director|"
    r"Emeritus|Emerita|Senior|Adjunct|Visiting|Status-Only|Teaching Stream|"
    r"Research Stream|University Professor|Dean|Fellow)\b",
    re.IGNORECASE,
)

_robots_cache = {}
_last_request_time = {}

def is_allowed_by_robots(url: str) -> bool:
    parsed = urlparse(url)
    domain = f"{parsed.scheme}://{parsed.netloc}"

    if domain not in _robots_cache:
        rp = robotparser.RobotFileParser()
        rp.set_url(urljoin(domain, "/robots.txt"))

        try:
            rp.read()
        except Exception:
            log.warning(
                "Could not read robots.txt for %s — proceeding cautiously.",
                domain
            )
            _robots_cache[domain] = None
            return True

        _robots_cache[domain] = rp

    rp = _robots_cache[domain]

    if rp is None:
        return True

    return rp.can_fetch(
        HEADERS["User-Agent"],
        url
    )

def polite_get(url: str):
    domain = urlparse(url).netloc

    if not is_allowed_by_robots(url):
        log.warning(
            "Blocked by robots.txt, skipping: %s",
            url
        )
        return None

    last = _last_request_time.get(domain, 0)
    wait = REQUEST_DELAY_SECONDS - (
        time.time() - last
    )

    if wait > 0:
        time.sleep(wait)

    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):
        try:
            resp = requests.get(
                url,
                headers=HEADERS,
                timeout=REQUEST_TIMEOUT
            )

            _last_request_time[domain] = time.time()

            if resp.status_code == 200:
                return resp

            log.warning(
                "Status %s for %s (attempt %d)",
                resp.status_code,
                url,
                attempt
            )

        except requests.RequestException as e:
            log.warning(
                "Request error for %s (attempt %d): %s",
                url,
                attempt,
                e
            )

        time.sleep(1.5 * attempt)

    return None

def clean_name(text: str) -> str:
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip(" \t\n\r,;:-")
    return text

def looks_like_name(text: str) -> bool:
    if not text or len(text) > 60 or len(text) < 3:
        return False

    lowered = text.lower()

    if any(
        w == lowered
        for w in IGNORE_NAME_WORDS
    ):
        return False

    if "@" in text or "http" in lowered:
        return False

    words = text.split()

    if not (1 < len(words) <= 5):
        return False

    return True

def strip_title_words(text: str) -> str:
    match = TITLE_KEYWORDS_REGEX.search(text)

    if match:
        text = text[:match.start()]

    return clean_name(text)

def extract_name_from_cell(cell) -> str:
    a = cell.find("a")

    if a:
        candidate = strip_title_words(
            clean_name(a.get_text())
        )

        if looks_like_name(candidate):
            return candidate

    candidate = strip_title_words(
        clean_name(cell.get_text())
    )

    if looks_like_name(candidate):
        return candidate

    return ""

def extract_name_near_email(
    soup_element,
    email: str
) -> str:

    tr = soup_element.find_parent("tr")

    if tr is not None:
        email_cell = soup_element.find_parent(
            ["td", "th"]
        )

        for cell in tr.find_all(
            ["td", "th"]
        ):
            if cell is email_cell:
                continue

            candidate = extract_name_from_cell(cell)

            if candidate:
                return candidate

    link_text = clean_name(
        soup_element.get_text()
    )

    if (
        looks_like_name(link_text)
        and "@" not in link_text
    ):
        return link_text

    parent = soup_element.parent
    depth = 0

    while parent is not None and depth < 5:

        for tag in parent.find_all(
            ["h1", "h2", "h3", "h4", "strong", "b"]
        ):
            candidate = clean_name(
                tag.get_text()
            )

            if looks_like_name(candidate):
                return candidate

        for tag in parent.find_all(
            class_=re.compile(r"name", re.I)
        ):
            candidate = clean_name(
                tag.get_text()
            )

            if looks_like_name(candidate):
                return candidate

        parent = parent.parent
        depth += 1

    return ""

def scrape_page(url: str):
    resp = polite_get(url)

    if resp is None:
        return []

    soup = BeautifulSoup(
        resp.text,
        "lxml"
    )

    results = []
    seen_emails = set()

    for a in soup.find_all(
        "a",
        href=True
    ):
        href = a["href"]

        if href.lower().startswith("mailto:"):

            email = (
                href
                .split(":", 1)[1]
                .split("?")[0]
                .strip()
            )

            if not EMAIL_REGEX.fullmatch(email):
                continue

            if any(
                bad in email.lower()
                for bad in IGNORE_EMAIL_DOMAINS_SUBSTRINGS
            ):
                continue

            if email.lower() in seen_emails:
                continue

            seen_emails.add(
                email.lower()
            )

            name = extract_name_near_email(
                a,
                email
            )

            results.append({
                "name": name,
                "email": email,
                "source_url": url
            })

    for match in EMAIL_REGEX.finditer(
        soup.get_text(" ")
    ):
        email = match.group(0)

        if email.lower() in seen_emails:
            continue

        if any(
            bad in email.lower()
            for bad in IGNORE_EMAIL_DOMAINS_SUBSTRINGS
        ):
            continue

        seen_emails.add(
            email.lower()
        )

        results.append({
            "name": "",
            "email": email,
            "source_url": url
        })

    return results

def scrape_university(entry: dict):
    rows = []

    for url in entry["faculty_urls"]:
        log.info(
            "Scraping %s — %s",
            entry["name"],
            url
        )

        found = scrape_page(url)

        for item in found:
            rows.append({
                "Name": item["name"],
                "Email": item["email"],
                "University": entry["name"],
                "Country": entry["country"],
                "Field": entry["field"],
            })

        log.info(
            "  -> found %d email(s)",
            len(found)
        )

    return rows

def main():
    all_rows = []

    for entry in UNIVERSITIES:
        rows = scrape_university(entry)
        all_rows.extend(rows)

    df = pd.DataFrame(
        all_rows,
        columns=[
            "Name",
            "Email",
            "University",
            "Country",
            "Field"
        ]
    )

    if df.empty:
        log.warning(
            "No results found. Check your faculty_urls / site structures."
        )
        return

    df = df.drop_duplicates(
        subset=["Email"]
    ).sort_values(
        ["Country", "University", "Name"]
    )

    out_path = "professors.xlsx"

    df.to_excel(
        out_path,
        index=False,
        engine="openpyxl"
    )

    abs_path = os.path.abspath(
        out_path
    )

    log.info(
        "Done. Wrote %d rows to %s",
        len(df),
        abs_path
    )

    print(
        f"\nExcel file written to: {abs_path}\n"
    )

if __name__ == "__main__":
    main()

