"""TMDB API calls and caching.

This module is the only part of the app that talks to The Movie Database
(TMDB) website. It requests movie data (titles, genres, overviews, cast,
keywords, streaming availability, etc.) and saves the results into
data/movies.json, so the app doesn't have to re-download the same
information every time it runs.

Main entry points:
    load_movies()  - what the rest of the app calls. Reads the cache file if
                     it exists, otherwise builds it from the API first.
    build_cache()  - downloads the movies and writes the cache file.
"""

import json
import os
import time
from pathlib import Path

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Settings (all in one place so they are easy to find and explain)
# ---------------------------------------------------------------------------
BASE_URL = "https://api.themoviedb.org/3"
CACHE_PATH = Path(__file__).parent / "data" / "movies.json"

TARGET_MOVIES = 1000   # how many movies we want in the corpus
MAX_PAGES = 100        # safety stop: never request more list pages than this

REQUEST_DELAY = 0.05   # seconds to pause after every successful request
MAX_RETRIES = 5        # how many times to try one request before giving up
BACKOFF_BASE = 1.0     # first retry waits 1s, then 2s, 4s, 8s, 16s
MAX_WAIT = 60          # never sleep longer than this on one retry (seconds)
REQUEST_TIMEOUT = 15   # give up on a single request after this many seconds

WATCH_REGION = "CA"                # whose streaming providers we keep
CERT_COUNTRIES = ("CA", "US")      # certification: try Canada first, then US


class TMDBError(Exception):
    """A request failed in a way we couldn't recover from."""


class TMDBAuthError(TMDBError):
    """TMDB rejected our token. Retrying won't help, so we stop everything."""


# ---------------------------------------------------------------------------
# Talking to the API
# ---------------------------------------------------------------------------
def _get_token():
    """Read TMDB_TOKEN from .streamlit/secrets.toml via Streamlit."""
    try:
        token = st.secrets["TMDB_TOKEN"]
    except Exception as exc:  # missing file or missing key
        raise RuntimeError(
            "Could not read TMDB_TOKEN from .streamlit/secrets.toml. "
            "Run from the project folder and make sure the file has a "
            'line like: TMDB_TOKEN = "..."'
        ) from exc
    if not token or token == "your-tmdb-token-here":
        raise RuntimeError("TMDB_TOKEN in secrets.toml is still the placeholder.")
    return token


def _make_session():
    """Create a requests Session that sends the Bearer token on every call.

    A Session reuses the same network connection between requests, which is
    noticeably faster than opening a new connection for each of ~1000 calls.
    """
    session = requests.Session()
    session.headers.update(
        {
            "Authorization": f"Bearer {_get_token()}",
            "accept": "application/json",
        }
    )
    return session


def _retry_after_seconds(response, fallback):
    """How long TMDB asks us to wait after a 429, falling back to our backoff."""
    try:
        return min(float(response.headers["Retry-After"]), MAX_WAIT)
    except (KeyError, ValueError):
        return min(fallback, MAX_WAIT)


def _get(session, path, params=None):
    """GET one TMDB endpoint and return its JSON (a dict).

    This function contains all of our rate-limit and error handling:
      * after every successful call we sleep REQUEST_DELAY (politeness)
      * 429 "too many requests" -> wait as long as TMDB asks, then retry
      * 5xx server errors, timeouts, dropped connections -> wait with
        exponential backoff (1s, 2s, 4s, ...) and retry
      * 404 (movie doesn't exist) -> return None so the caller can skip it
      * 401 (bad token) -> raise TMDBAuthError immediately; retrying is pointless
      * still failing after MAX_RETRIES attempts -> raise TMDBError
    """
    url = f"{BASE_URL}{path}"
    problem = "unknown error"

    for attempt in range(MAX_RETRIES):
        backoff = BACKOFF_BASE * 2**attempt

        try:
            response = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
        except (requests.ConnectionError, requests.Timeout) as exc:
            problem = f"network error: {exc}"
            time.sleep(min(backoff, MAX_WAIT))
            continue

        if response.status_code == 429:
            problem = "rate limited (HTTP 429)"
            time.sleep(_retry_after_seconds(response, backoff))
            continue
        if response.status_code >= 500:
            problem = f"server error (HTTP {response.status_code})"
            time.sleep(min(backoff, MAX_WAIT))
            continue
        if response.status_code == 404:
            return None
        if response.status_code == 401:
            raise TMDBAuthError("TMDB rejected the token (HTTP 401). Check secrets.toml.")
        if not response.ok:
            raise TMDBError(f"{path}: unexpected HTTP {response.status_code}")

        try:
            data = response.json()
        except ValueError:
            problem = "response was not valid JSON"
            time.sleep(min(backoff, MAX_WAIT))
            continue

        time.sleep(REQUEST_DELAY)
        return data

    raise TMDBError(f"Gave up on {path} after {MAX_RETRIES} attempts: {problem}")


# ---------------------------------------------------------------------------
# Step 1: decide WHICH movies to download
# ---------------------------------------------------------------------------
def _collect_movie_ids(session, target):
    """Walk the 'popular' and 'top rated' lists page by page, collecting
    unique movie IDs until we have `target` of them.

    The two lists overlap (a movie can be both popular and top rated), so we
    de-duplicate with a set and keep going until we have enough *unique* IDs.
    Each page holds 20 movies.
    """
    ids = []
    seen = set()

    for page in range(1, MAX_PAGES + 1):
        for list_name in ("popular", "top_rated"):
            data = _get(session, f"/movie/{list_name}", {"page": page})
            for item in (data or {}).get("results") or []:
                movie_id = item.get("id")
                if movie_id is not None and movie_id not in seen:
                    seen.add(movie_id)
                    ids.append(movie_id)
        if len(ids) >= target:
            break

    return ids[:target]


# ---------------------------------------------------------------------------
# Step 2: turn one raw TMDB response into a tidy dictionary
# ---------------------------------------------------------------------------
def _names(items, key="name", limit=None):
    """Pull one field (usually 'name') out of a list of dicts, skipping gaps.

    Watch providers are the odd one out: TMDB calls their field
    'provider_name', so we pass key="provider_name" for those.
    """
    names = [i[key] for i in items or [] if i.get(key)]
    return names[:limit] if limit else names


def _pick_certification(raw):
    """Find an age rating (e.g. 'PG', '14A'), preferring Canada over the US.

    Returns (certification, country). Both are "" if nothing was found.
    """
    by_country = {
        entry.get("iso_3166_1"): entry.get("release_dates") or []
        for entry in (raw.get("release_dates") or {}).get("results") or []
    }
    for country in CERT_COUNTRIES:
        for release in by_country.get(country, []):
            if release.get("certification"):
                return release["certification"], country
    return "", ""


def _parse_movie(raw):
    """Keep only the fields we need. Every lookup tolerates missing data."""
    release_date = raw.get("release_date") or ""
    year = int(release_date[:4]) if release_date[:4].isdigit() else None

    credits = raw.get("credits") or {}
    cast = sorted(credits.get("cast") or [], key=lambda c: c.get("order", 9999))
    directors = [
        c["name"]
        for c in credits.get("crew") or []
        if c.get("job") == "Director" and c.get("name")
    ]

    certification, cert_country = _pick_certification(raw)

    providers = (
        ((raw.get("watch/providers") or {}).get("results") or {}).get(WATCH_REGION)
        or {}
    )

    return {
        "id": raw.get("id"),
        "title": raw.get("title") or "",
        "original_language": raw.get("original_language") or "",  # ISO code, e.g. "en"
        "release_year": year,
        "runtime": raw.get("runtime") or None,  # TMDB uses 0 for "unknown"
        "certification": certification,
        "certification_country": cert_country,
        "genres": _names(raw.get("genres")),
        "overview": raw.get("overview") or "",
        "keywords": _names((raw.get("keywords") or {}).get("keywords")),
        "cast": _names(cast, limit=5),  # cast is already sorted by billing order
        "director": ", ".join(directors),
        "vote_average": raw.get("vote_average"),
        "vote_count": raw.get("vote_count"),
        "providers_ca": {
            "stream": _names(providers.get("flatrate"), key="provider_name"),
            "rent": _names(providers.get("rent"), key="provider_name"),
            "buy": _names(providers.get("buy"), key="provider_name"),
        },
        "poster_path": raw.get("poster_path") or "",
    }


# ---------------------------------------------------------------------------
# Step 3: download everything and save it
# ---------------------------------------------------------------------------
def build_cache(target=TARGET_MOVIES, path=CACHE_PATH):
    """Download ~`target` movies from TMDB and save them to `path` as JSON."""
    session = _make_session()

    print("Collecting movie IDs from the popular and top-rated lists...")
    movie_ids = _collect_movie_ids(session, target)
    print(f"Found {len(movie_ids)} unique movies. Fetching details...")

    # One request per movie. append_to_response asks TMDB to bundle the
    # credits, keywords, release dates (certifications) and watch providers
    # into the same reply as the details, instead of 4 separate requests.
    extras = "credits,keywords,release_dates,watch/providers"

    movies = []
    skipped = 0
    started = time.time()
    for n, movie_id in enumerate(movie_ids, start=1):
        try:
            raw = _get(session, f"/movie/{movie_id}", {"append_to_response": extras})
        except TMDBAuthError:
            raise  # bad token: every later request would fail too
        except TMDBError as exc:
            print(f"  skipping movie {movie_id}: {exc}")
            raw = None

        if raw is None:
            skipped += 1
        else:
            movies.append(_parse_movie(raw))

        if n % 50 == 0 or n == len(movie_ids):
            print(f"  {n}/{len(movie_ids)} done ({time.time() - started:.0f}s elapsed)")

    _write_cache(movies, path)
    print(f"Saved {len(movies)} movies to {path} ({skipped} skipped).")
    return movies


def _write_cache(movies, path):
    """Save the movie list as JSON.

    Writes to a temporary file first, then swaps it in. That way a crash
    mid-write can never leave a half-written movies.json behind.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)  # data/ is git-ignored
    temp_path = path.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(movies, f, ensure_ascii=False, indent=2)
    os.replace(temp_path, path)


def add_original_language(path=CACHE_PATH):
    """One-off upgrade for caches built before original_language was saved.

    Fetches just that one field for each cached movie (about 1000 quick
    requests, ~3 minutes) and writes it into the existing cache file, so the
    set of movies stays exactly the same. Movies that already have the field
    are skipped, so it is safe to run twice.
    """
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        movies = json.load(f)
    todo = [m for m in movies if "original_language" not in m]
    if not todo:
        print("Every movie already has original_language.")
        return movies

    session = _make_session()
    print(f"Fetching original_language for {len(todo)} movies...")
    failed = 0
    for n, movie in enumerate(todo, start=1):
        try:
            raw = _get(session, f"/movie/{movie['id']}")
        except TMDBAuthError:
            raise
        except TMDBError as exc:
            print(f"  skipping movie {movie['id']}: {exc}")
            raw = None
        if raw is None:
            failed += 1
        else:
            movie["original_language"] = raw.get("original_language") or ""
        if n % 100 == 0 or n == len(todo):
            print(f"  {n}/{len(todo)} done")

    _write_cache(movies, path)
    print(f"Saved. {failed} movies could not be updated (run again to retry them).")
    return movies


def load_movies(path=CACHE_PATH):
    """Return the list of movies, using the cache file when it exists.

    Only calls the API (via build_cache) if the cache file is missing.
    """
    path = Path(path)
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return build_cache(path=path)
