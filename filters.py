"""The filtering logic.

This module narrows down the list of movies based on what the user asks
for. It removes movies that don't fit *before* they are ranked: the ids it
returns are passed to recommender.recommend(candidate_ids=...).

Main entry points:
    apply_filters(movies, **criteria)  - ids of the movies that pass every filter
    explain_filters(movies, **criteria) - the count, plus a plain-language reason
                                          and suggestions when few or no movies match
    available_genres / _languages / _platforms / _decades / _certifications
                                       - (value, label, count) lists for UI dropdowns

The filters (all optional):
    Temporal      year_from, year_to, decades
    Quality       min_rating, min_votes
    Content       min_runtime (default 60), max_runtime, certifications, languages
    Personnel     person (an actor or director name), person_role
    Genres        genres_all (AND), genres_any (OR), genres_none (NOT)
    Streaming     platforms (available to stream in Canada)

How filters combine:
  * Different filters are always ANDed: a movie must pass every active one.
  * Several values inside one filter (e.g. two languages, two platforms) are
    ORed: the movie may match any of them. genres_all is the exception: it
    needs every genre listed.
  * A filter that isn't set does nothing. Because every filter is just a
    yes/no test on a movie, they can't conflict in how they run. Some
    *selections* can never be satisfied together (a genre both required and
    excluded, years from 2010 to 2000); those return no movies and are
    reported in plain language by explain_filters().
  * Missing data: range filters (years, rating, votes, runtime) only remove a
    movie they can show is outside the range, so unknown values are kept.
    "Must be X" filters (certification, language, genre, person, platform)
    need evidence of a match, so unknown values fail.
"""

import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

import nlp

# Anything shorter than this is treated as a short or TV special (30 of the
# 1000 cached movies). The UI can turn the rule off with min_runtime=None.
DEFAULT_MIN_RUNTIME = 60

# explain_filters() writes an explanation whenever fewer movies than this match.
FEW_RESULTS = 10

PERSON_ROLES = ("any", "actor", "director")

FILTER_LABELS = {
    "release_year": "Release year",
    "decade": "Decade",
    "rating": "Minimum rating",
    "votes": "Minimum votes",
    "runtime": "Runtime",
    "certification": "Certification",
    "language": "Original language",
    "person": "Actor/director",
    "genres_all": "Genres (must have all)",
    "genres_any": "Genres (any of)",
    "genres_none": "Genres (excluding)",
    "platform": "Streaming platform",
}

# TMDB gives ISO language codes. 'cn' (Cantonese) and 'xx' (no spoken
# language) are TMDB-specific.
LANGUAGE_NAMES = {
    "en": "English", "ja": "Japanese", "es": "Spanish", "it": "Italian",
    "fr": "French", "ko": "Korean", "zh": "Mandarin", "cn": "Cantonese",
    "pt": "Portuguese", "de": "German", "ru": "Russian", "sv": "Swedish",
    "hi": "Hindi", "th": "Thai", "ta": "Tamil", "fa": "Persian",
    "tr": "Turkish", "ar": "Arabic", "ml": "Malayalam", "da": "Danish",
    "hu": "Hungarian", "tl": "Tagalog", "nl": "Dutch", "id": "Indonesian",
    "ro": "Romanian", "no": "Norwegian", "hr": "Croatian", "lv": "Latvian",
    "pl": "Polish", "kn": "Kannada", "bn": "Bengali", "te": "Telugu",
    "he": "Hebrew", "el": "Greek", "cs": "Czech", "fi": "Finnish",
    "uk": "Ukrainian", "vi": "Vietnamese", "ur": "Urdu", "is": "Icelandic",
    "xx": "No spoken language",
}
_LANGUAGE_CODES = {name.lower(): code for code, name in LANGUAGE_NAMES.items()}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _as_list(value):
    """None -> [], "x" -> ["x"], any collection -> list; blanks removed."""
    if value is None:
        return []
    if isinstance(value, (str, int)):
        value = [value]
    return [v for v in value if v is not None and str(v).strip() != ""]


def _decade_start(value):
    """1990, "1990", "1990s" or 1995 -> 1990. Anything else is an error."""
    text = str(value).strip().lower().rstrip("s")
    if not (text.isdigit() and len(text) == 4):
        raise ValueError(f"Unrecognised decade {value!r}; use e.g. 1990 or '1990s'.")
    return int(text) // 10 * 10


def _name_tokens(text):
    """'Penélope Cruz' -> ['penelope', 'cruz']: lowercase, accents removed."""
    text = unicodedata.normalize("NFKD", str(text or "").lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.findall(r"\w+", text)


def _name_matches(query_tokens, name):
    """Every typed word must be the start of some word in the name, so
    'nolan' and 'christopher nol' both find 'Christopher Nolan'."""
    name_tokens = _name_tokens(name)
    return all(any(t.startswith(q) for t in name_tokens) for q in query_tokens)


_PLATFORM_SUFFIX = re.compile(
    r"\s+(amazon channel|apple tv channel|standard with ads|basic with ads|with ads|premium)$",
    re.IGNORECASE,
)


def _strip_platform(name):
    """'Crave Amazon Channel' -> 'Crave'; 'Netflix Standard with Ads' -> 'Netflix'.

    TMDB lists one service several times: the service itself, the same service
    sold through an Amazon or Apple TV channel, and cheaper ad-supported tiers.
    For "is this movie on Crave?" those are all Crave.
    """
    name = name.strip()
    while True:
        shorter = _PLATFORM_SUFFIX.sub("", name)
        if shorter == name:
            return name
        name = shorter


def platform_key(name):
    """Comparison key for a platform name: variants and 'Disney+' / 'Disney Plus'
    spellings all map to the same key."""
    return re.sub(r"\s+", " ", _strip_platform(name).lower().replace("+", " plus")).strip()


def streaming_platforms(movie):
    """Platforms a movie streams on in Canada, one entry per brand.

    e.g. ["Crave", "Netflix"]. Variants are merged the same way the platform
    filter merges them, so what the UI shows matches what the filter matches.
    """
    brands = {}
    for name in (movie.get("providers_ca") or {}).get("stream") or []:
        brands.setdefault(platform_key(name), _strip_platform(name))
    return list(brands.values())


def _stream_keys(movie):
    stream = (movie.get("providers_ca") or {}).get("stream") or []
    return {platform_key(name) for name in stream}


def _in_range(value, low, high):
    if low is not None and high is not None and low > high:
        return False  # backwards range: nothing fits, not even unknown values
    if value is None:  # unknown: can't show it's outside the range, so keep it
        return True
    return (low is None or value >= low) and (high is None or value <= high)


# ---------------------------------------------------------------------------
# The filter settings
# ---------------------------------------------------------------------------
@dataclass
class Filters:
    """All filter settings. Every one is optional; None/empty means "no filter"."""

    year_from: int | None = None          # inclusive
    year_to: int | None = None            # inclusive
    decades: list | None = None           # e.g. [1980, "1990s"]
    min_rating: float | None = None       # TMDB average vote, 0-10
    min_votes: int | None = None
    min_runtime: int | None = DEFAULT_MIN_RUNTIME   # minutes; None turns it off
    max_runtime: int | None = None        # minutes, inclusive. "Under 2 hours" = 119
    certifications: list | None = None    # "family", "teen", "adult", "unrated"
    languages: list | None = None         # codes ("fr") or names ("French")
    person: str | None = None             # actor or director name
    person_role: str = "any"              # "any", "actor" or "director"
    genres_all: list | None = None        # movie must have ALL of these
    genres_any: list | None = None        # movie must have AT LEAST ONE
    genres_none: list | None = None       # movie must have NONE of these
    platforms: list | None = None         # on any of these, streaming in Canada

    def __post_init__(self):
        # Mistyped option values are bugs in the calling code, so they raise
        # an error at once rather than silently matching nothing.
        self.decades = [_decade_start(d) for d in _as_list(self.decades)]
        self.certifications = [str(c).strip().lower() for c in _as_list(self.certifications)]
        unknown = [c for c in self.certifications if c not in nlp.BUCKETS]
        if unknown:
            raise ValueError(f"Unknown certification bucket(s) {unknown}; use {nlp.BUCKETS}.")
        self.person_role = str(self.person_role).strip().lower()
        if self.person_role not in PERSON_ROLES:
            raise ValueError(f"person_role must be one of {PERSON_ROLES}, not {self.person_role!r}.")
        self.languages = [
            _LANGUAGE_CODES.get(str(v).strip().lower(), str(v).strip().lower())
            for v in _as_list(self.languages)
        ]
        self.genres_all = [str(g).strip() for g in _as_list(self.genres_all)]
        self.genres_any = [str(g).strip() for g in _as_list(self.genres_any)]
        self.genres_none = [str(g).strip() for g in _as_list(self.genres_none)]
        self.platforms = [str(p).strip() for p in _as_list(self.platforms)]
        self.person = (self.person or "").strip() or None

    # -- the yes/no tests ----------------------------------------------------
    def tests(self):
        """{filter name: function(movie) -> True/False} for the active filters."""
        tests = {}

        if self.year_from is not None or self.year_to is not None:
            tests["release_year"] = lambda m: _in_range(m.get("release_year"), self.year_from, self.year_to)

        if self.decades:
            decades = set(self.decades)
            tests["decade"] = lambda m: (
                m.get("release_year") is None or m["release_year"] // 10 * 10 in decades
            )

        if self.min_rating is not None:
            tests["rating"] = lambda m: _in_range(m.get("vote_average"), self.min_rating, None)

        if self.min_votes is not None:
            tests["votes"] = lambda m: _in_range(m.get("vote_count"), self.min_votes, None)

        if self.min_runtime is not None or self.max_runtime is not None:
            tests["runtime"] = lambda m: _in_range(m.get("runtime"), self.min_runtime, self.max_runtime)

        # Each test below keeps its own uniquely named variable. (Sharing one
        # name such as `wanted` would make all the tests see the last value.)
        if self.certifications:
            wanted_buckets = set(self.certifications)
            tests["certification"] = lambda m: nlp.certification_bucket(m.get("certification")) in wanted_buckets

        if self.languages:
            wanted_languages = set(self.languages)
            tests["language"] = lambda m: (m.get("original_language") or "").lower() in wanted_languages

        if self.person:
            query = _name_tokens(self.person)
            role = self.person_role

            def person_test(m):
                names = []
                if role in ("any", "actor"):
                    names += m.get("cast") or []
                if role in ("any", "director"):
                    names += (m.get("director") or "").split(",")
                return any(_name_matches(query, n) for n in names)

            if query:
                tests["person"] = person_test

        genres = lambda m: {g.lower() for g in m.get("genres") or []}
        if self.genres_all:
            need = {g.lower() for g in self.genres_all}
            tests["genres_all"] = lambda m: need <= genres(m)
        if self.genres_any:
            any_of = {g.lower() for g in self.genres_any}
            tests["genres_any"] = lambda m: bool(any_of & genres(m))
        if self.genres_none:
            banned = {g.lower() for g in self.genres_none}
            tests["genres_none"] = lambda m: not (banned & genres(m))

        if self.platforms:
            wanted_platforms = {platform_key(p) for p in self.platforms}
            tests["platform"] = lambda m: bool(wanted_platforms & _stream_keys(m))

        return tests

    # -- selections that can never be satisfied together ----------------------
    def conflicts(self):
        """Plain-language descriptions of settings that contradict each other."""
        found = []
        low, high = self.year_from, self.year_to
        if low is not None and high is not None and low > high:
            found.append(f"The release-year range is backwards (from {low} to {high}).")
        if self.decades and (low is not None or high is not None):
            lo = low if low is not None else -10**9
            hi = high if high is not None else 10**9
            if not any(d <= hi and d + 9 >= lo for d in self.decades):
                found.append("The selected decade(s) fall outside the chosen year range.")
        if self.min_runtime is not None and self.max_runtime is not None and self.min_runtime > self.max_runtime:
            hint = ""
            if self.min_runtime == DEFAULT_MIN_RUNTIME:
                hint = f" The default {DEFAULT_MIN_RUNTIME}-minute minimum hides shorts; turn it off to include them."
            found.append(f"The minimum runtime ({self.min_runtime} min) is longer than the maximum ({self.max_runtime} min).{hint}")
        required = {g.lower(): g for g in self.genres_all}
        banned = {g.lower(): g for g in self.genres_none}
        for key in required.keys() & banned.keys():
            found.append(f"The genre '{required[key]}' is both required and excluded.")
        any_of = {g.lower() for g in self.genres_any}
        if any_of and any_of <= banned.keys():
            found.append("Every genre in 'any of' is also in the excluded list.")
        return found

    def unknown_values(self, movies):
        """Selected values that no movie in the catalogue has (e.g. a language
        or platform that doesn't appear in the data)."""
        found = []
        have_languages = {(m.get("original_language") or "").lower() for m in movies}
        for code in self.languages:
            if code not in have_languages:
                found.append(f"No movie in the catalogue has the original language '{LANGUAGE_NAMES.get(code, code)}'.")
        have_platforms = set().union(*(_stream_keys(m) for m in movies)) if movies else set()
        for name in self.platforms:
            if platform_key(name) not in have_platforms:
                found.append(f"No movie in the catalogue streams on '{name}' in Canada.")
        have_genres = {g.lower() for m in movies for g in m.get("genres") or []}
        for genre in self.genres_all + self.genres_any + self.genres_none:
            if genre.lower() not in have_genres:
                found.append(f"No movie in the catalogue has the genre '{genre}'.")
        return found


# ---------------------------------------------------------------------------
# Applying the filters
# ---------------------------------------------------------------------------
def _passes(settings, movies):
    """{filter name: array of True/False per movie} for each active filter."""
    return {
        name: np.array([bool(test(m)) for m in movies], dtype=bool)
        for name, test in settings.tests().items()
    }


def apply_filters(movies, **criteria):
    """Return the ids of the movies that pass every filter, in catalogue order.

    Takes any of the keyword settings listed in the module docstring (see the
    Filters class for exact types), e.g.
        apply_filters(movies, decades=[1990], genres_all=["Comedy"],
                      genres_none=["Horror"], platforms=["Netflix", "Crave"])
    The 60-minute minimum runtime is on unless you pass min_runtime=None.
    A mistyped setting name raises TypeError. An impossible combination simply
    returns []; explain_filters() says why.
    """
    keep = np.ones(len(movies), dtype=bool)
    for passed in _passes(Filters(**criteria), movies).values():
        keep &= passed
    return [movie["id"] for movie, ok in zip(movies, keep) if ok]


def explain_filters(movies, **criteria):
    """Count the matches and explain, in plain language, why there are few or none.

    Takes the same settings as apply_filters. Returns a dict:
        count          number of movies that pass every filter
        conflicts      settings that contradict each other (see Filters.conflicts)
        unknown_values selected values no movie has
        alone          {filter: movies passing that one filter by itself}
        if_removed     {filter: movies that would match if only that filter were dropped}
        suggestions    [{"filter", "label", "count_if_removed"}], best first
        message        ready-to-show text, or None when count >= FEW_RESULTS
    """
    settings = Filters(**criteria)
    passed = _passes(settings, movies)
    names = list(passed)
    size = len(movies)

    def combined(skip=None):
        keep = np.ones(size, dtype=bool)
        for name in names:
            if name != skip:
                keep &= passed[name]
        return keep

    count = int(combined().sum())
    alone = {name: int(passed[name].sum()) for name in names}
    if_removed = {name: int(combined(skip=name).sum()) for name in names}
    conflicts = settings.conflicts()
    unknown = settings.unknown_values(movies)
    suggestions = [
        {"filter": name, "label": FILTER_LABELS[name], "count_if_removed": gain}
        for name, gain in sorted(if_removed.items(), key=lambda kv: -kv[1])
        if gain > count
    ]

    message = None
    if count < FEW_RESULTS:
        message = ("No movies match your filters." if count == 0
                   else f"Only {count} movie{'' if count == 1 else 's'} match{'es' if count == 1 else ''} your filters.")
        if conflicts:
            message += " Some settings contradict each other: " + " ".join(conflicts)
        elif unknown:
            message += " " + " ".join(unknown)
        else:
            empty = [FILTER_LABELS[n] for n in names if alone[n] == 0]
            if empty:
                message += " Nothing in the catalogue matches " + " or ".join(empty) + " even on its own."
        if suggestions:
            message += " Relaxing a filter would help: " + "; ".join(
                f"dropping '{s['label']}' gives {s['count_if_removed']}" for s in suggestions[:3]
            ) + "."

    return {
        "count": count,
        "conflicts": conflicts,
        "unknown_values": unknown,
        "alone": alone,
        "if_removed": if_removed,
        "suggestions": suggestions,
        "message": message,
    }


# ---------------------------------------------------------------------------
# Lists for UI dropdowns: (value to pass to apply_filters, label to show, movie count)
# ---------------------------------------------------------------------------
def available_genres(movies):
    counts = Counter(g for m in movies for g in set(m.get("genres") or []))
    return [(g, g, n) for g, n in counts.most_common()]


def available_languages(movies):
    counts = Counter((m.get("original_language") or "") for m in movies)
    return [(code, LANGUAGE_NAMES.get(code, code), n) for code, n in counts.most_common() if code]


def available_decades(movies):
    counts = Counter(m["release_year"] // 10 * 10 for m in movies if m.get("release_year") is not None)
    return [(d, f"{d}s", counts[d]) for d in sorted(counts)]


def available_certifications(movies):
    counts = Counter(nlp.certification_bucket(m.get("certification")) for m in movies)
    return [(b, b.title(), counts[b]) for b in nlp.BUCKETS]


def available_platforms(movies):
    """Streaming platforms (variants merged), most movies first. Pass the
    value or label straight to apply_filters(platforms=[...])."""
    movie_counts = Counter()
    spellings = defaultdict(Counter)
    for m in movies:
        names = (m.get("providers_ca") or {}).get("stream") or []
        movie_counts.update({platform_key(n) for n in names})
        for n in names:
            spellings[platform_key(n)][_strip_platform(n)] += 1
    return [
        (spellings[key].most_common(1)[0][0],) * 2 + (n,)
        for key, n in movie_counts.most_common()
    ]
