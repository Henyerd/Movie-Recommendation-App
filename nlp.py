"""TF-IDF features and VADER sentiment.

NLP means "natural language processing": getting a computer to work with
human text. This module turns each movie into numbers in three ways:

  1. A TF-IDF vector built from genres, keywords, cast and director, so any
     two movies (or a typed query and a movie) can be compared for similarity.
  2. A VADER sentiment score for the plot overview (how positive or negative
     the wording sounds, from -1 to +1).
  3. A certification bucket (family / teen / adult / unrated).

Main entry points:
    get_features(movies)         - computes everything once and caches it
    query_scores(text, features) - scores every movie by how many of the typed
                                   query's words it matches (used for queries)
    query_vector(text, features) - turns a typed query into a vector in the
                                   same space as the movies
    similarity_matrix(features)  - movie-to-movie similarity scores
    certification_bucket(code)   - "PG-13" / "14A" -> "teen", etc.
"""

import re
from dataclasses import dataclass

import numpy as np
import streamlit as st
from scipy import sparse
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import normalize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
# How much each field contributes to the similarity score. These add up to 1,
# so read them as percentages: 40% of "how similar are these two movies" comes
# from genre agreement, 35% from keywords, 15% from cast, 10% from director.
# The one-letter codes are prefixes we attach to each word (see _movie_text).
#
# Tested and kept: we tried genre 40 / keywords 40 / cast 15 / director 5 and
# compared nearest neighbours on The Dark Knight, Toy Story, Spirited Away and
# The Godfather. Results were mostly unchanged (92% of all 1000 movies kept the
# same #1 neighbour), but the one real change was a small loss on a
# same-director match: Castle in the Sky (also by Miyazaki) dropped out of
# Spirited Away's top 5. So director stays at 10%. This was a by-eye check on a
# handful of films with no ground truth, not a proven improvement either way.
FIELD_WEIGHTS = {"g": 0.40, "k": 0.35, "c": 0.15, "d": 0.10}

# Informal words people type -> the genre word TF-IDF can actually match.
# TF-IDF only matches identical words; it has no idea "funny" means "comedy".
QUERY_SYNONYMS = {
    "funny": ["comedy"], "hilarious": ["comedy"], "humorous": ["comedy"],
    "laugh": ["comedy"], "laughs": ["comedy"], "silly": ["comedy"],
    "scary": ["horror"], "creepy": ["horror"], "spooky": ["horror"],
    "terrifying": ["horror"], "frightening": ["horror"],
    "romantic": ["romance"], "love": ["romance"],
    "exciting": ["action", "thriller"], "thrilling": ["thriller"],
    "suspenseful": ["thriller"], "tense": ["thriller"],
    "emotional": ["drama"], "sad": ["drama"], "cartoon": ["animation"],
    "animated": ["animation"], "kids": ["family"], "children": ["family"],
    "fantastical": ["fantasy"], "magical": ["fantasy"],
}

# "not X", "no X", "without X", "never X" (optionally "not too X", etc.)
_NEGATION = re.compile(
    r"\b(?:not|no|without|never)\s+(?:(?:too|so|very|overly|that)\s+)?[a-z']+"
)

# Words that say nothing about taste in a query. "movie" in particular would
# otherwise match the genre "TV Movie" and pull in unrelated films.
_QUERY_FILLER = {"movie", "movies", "film", "films", "flick", "flicks", "watch"}

_ANALYZER = SentimentIntensityAnalyzer()

BUCKETS = ("family", "teen", "adult", "unrated")


# ---------------------------------------------------------------------------
# Part 1: TF-IDF content features
# ---------------------------------------------------------------------------
def _words(text):
    """Lowercase text -> list of meaningful words (stop words like 'the' dropped)."""
    return [
        w
        for w in re.findall(r"[a-z0-9]+", (text or "").lower())
        if len(w) > 1 and w not in ENGLISH_STOP_WORDS
    ]


def _person_token(name):
    """'Christian Bale' -> 'christianbale'.

    Each person becomes ONE token. Otherwise two different people who share a
    first name ('Chris', 'Michael') would look like a match.
    """
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


def _movie_text(movie):
    """Combine genres, keywords, cast and director into one text string.

    Every word gets a one-letter field prefix, e.g.
        'g_comedy k_dystopia k_robot c_christianbale d_christophernolan'
    The prefix keeps the fields apart: the genre 'drama' and the keyword
    'drama' are different features, which lets us weight each field separately.
    Missing fields just contribute nothing, so nothing here can crash.
    """
    parts = []
    for genre in movie.get("genres") or []:
        parts += [f"g_{w}" for w in _words(genre)]
    for keyword in movie.get("keywords") or []:
        parts += [f"k_{w}" for w in _words(keyword)]
    for name in movie.get("cast") or []:
        if _person_token(name):
            parts.append(f"c_{_person_token(name)}")
    for name in (movie.get("director") or "").split(","):  # "A, B" if co-directed
        if _person_token(name):
            parts.append(f"d_{_person_token(name)}")
    return " ".join(parts)


def _apply_field_weights(X, col_fields):
    """Give each field its share of the similarity score (see FIELD_WEIGHTS).

    Steps, for each field (genre, keyword, cast, director):
      1. Take that field's columns and scale each movie's row so the field
         has length 1. Now a movie with 20 keywords and one with 3 count the
         same: keywords can't win just by being numerous.
      2. Multiply by sqrt(weight). Similarity is a dot product, so the weight
         gets applied twice (once per movie); the square root makes the final
         contribution exactly `weight`.
    The result: the dot product of two movies is exactly
        0.40*genre_sim + 0.35*keyword_sim + 0.15*cast_sim + 0.10*director_sim
    where each *_sim is a cosine similarity (0 to 1) within that field.

    We deliberately do NOT rescale rows afterwards. A field a movie lacks (e.g.
    no keywords) stays all zeros and simply counts as "no agreement" on that
    field. Rescaling would blow up the remaining fields and make sparse movies
    look artificially similar to everything.
    """
    blocks, order = [], []
    for field, weight in FIELD_WEIGHTS.items():
        cols = np.flatnonzero(col_fields == field)
        if cols.size:
            blocks.append(normalize(X[:, cols]) * np.sqrt(weight))
            order.append(cols)
    combined = sparse.hstack(blocks, format="csr")
    return combined[:, np.argsort(np.concatenate(order))]  # restore column order


def _query_groups(query):
    """Split a typed query into one group of alternative words per typed word.

    "something funny" -> [["funny", "comedy"]]: the typed word plus any
    synonyms from QUERY_SYNONYMS. A movie matching ANY word in a group counts
    as matching that typed word.
    """
    q = (query or "").lower().replace("sci-fi", "science fiction").replace("scifi", "science fiction")
    q = _NEGATION.sub(" ", q)  # drop negated words: "not stupid" -> ""
    return [
        [w] + QUERY_SYNONYMS.get(w, [])
        for w in _words(q)
        if w not in _QUERY_FILLER
    ]


def _query_text(query):
    """Turn a typed query into prefixed words in the same style as _movie_text."""
    words = [w for group in _query_groups(query) for w in group]
    # We don't know whether a typed word is a genre or a keyword, so offer it
    # as both. Words the model has never seen are ignored automatically.
    return " ".join([f"g_{w}" for w in words] + [f"k_{w}" for w in words])


def query_vector(query, features):
    """Turn free text like "something funny but not stupid" into a vector in
    the same space as the movies.

    NOTE: the recommender no longer ranks typed queries with this vector,
    because cosine similarity penalises movies that have extra genres (see
    query_scores below). It is kept for anyone who wants the raw TF-IDF
    vector. Returns a 1-row sparse matrix. `features.tfidf_matrix @ vec.T` gives the
    similarity of the query to every movie (0 to 1). Unlike the movies, the
    query IS rescaled to length 1: it only mentions the fields the user typed
    about, so a genre-only query like "funny" can still reach a score of 1 for a
    perfect genre match. Fields the query doesn't touch are ignored.
    If none of the query's words are known, the vector is all zeros (every
    similarity is 0) instead of an error.
    """
    raw = features.vectorizer.transform([_query_text(query)])
    return normalize(_apply_field_weights(raw, features.col_fields))


def query_scores(query, features):
    """Score every movie by how many of the query's words it matches (0 to 1).

    This is what the recommender uses for typed queries. Cosine similarity
    asks "are these two word-lists the same shape?", so a "Comedy, Adventure,
    Animation" movie scored far lower than a "Comedy"-only movie for the query
    "funny", even though both are funny. Here we ask a simpler question: of
    the words the user typed, what share does this movie have?

      * Each typed word (plus its synonyms) is one item. Words we have never
        seen in any movie's genres or keywords are ignored, not counted
        against the movie.
      * A movie matches an item if the word appears in its genres or its
        keywords. A genre match counts fully; a keyword match counts a little
        less, in the same ratio as FIELD_WEIGHTS (35/40 = 0.875).
      * The score is the average over items. "romantic comedy": a movie that
        is both scores 1.0, a comedy that isn't romantic scores 0.5.

    Extra genres on a movie never lower its score. Returns None if none of the
    query's words are recognised (so the caller can skip this component).
    """
    vocabulary = features.vectorizer.vocabulary_
    best_field = max(FIELD_WEIGHTS["g"], FIELD_WEIGHTS["k"])
    credit = {f: FIELD_WEIGHTS[f] / best_field for f in ("g", "k")}

    per_item = []
    for group in _query_groups(query):
        item = None
        for word in group:
            for field in ("g", "k"):
                column = vocabulary.get(f"{field}_{word}")
                if column is not None:
                    hit = features.presence[:, column].toarray().ravel() * credit[field]
                    item = hit if item is None else np.maximum(item, hit)
        if item is not None:
            per_item.append(item)
    return np.mean(per_item, axis=0) if per_item else None


# ---------------------------------------------------------------------------
# Part 2: VADER sentiment
# ---------------------------------------------------------------------------
def sentiment_score(text):
    """VADER compound score for a plot overview, from -1 (negative) to +1.

    We score each sentence and average, rather than scoring the whole
    overview in one go. VADER's compound score saturates toward +/-1 as text
    gets longer, which made 13% of movies look extreme on the raw whole-text
    score (135 of 1000 beyond +/-0.9); averaging sentences keeps scores
    spread out (33 of 1000).
    Empty text returns 0.0 (neutral), never an extreme value.
    """
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", (text or "").strip()) if s]
    if not sentences:
        return 0.0
    scores = [_ANALYZER.polarity_scores(s)["compound"] for s in sentences]
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Part 3: certification buckets
# ---------------------------------------------------------------------------
# Canadian and US codes mostly share the same letters, so one table covers both.
_FAMILY = {"G", "PG", "8+"}
_TEEN = {"PG-13", "13+", "14A", "14+", "16+"}
_ADULT = {"R", "NC-17", "18A", "18+", "A", "X"}


def certification_bucket(certification):
    """Collapse a Canadian or US rating into 'family', 'teen' or 'adult'.

    Missing, blank, 'NR' (not rated) and any code we don't recognise return
    'unrated', so those movies are kept rather than dropped.
    """
    code = (certification or "").strip().upper()
    if code in _FAMILY:
        return "family"
    if code in _TEEN:
        return "teen"
    if code in _ADULT:
        return "adult"
    return "unrated"


# ---------------------------------------------------------------------------
# Putting it together (and caching it)
# ---------------------------------------------------------------------------
@dataclass
class NLPFeatures:
    """Everything computed from the movie list. Row i always matches movies[i]."""

    movie_ids: list
    vectorizer: TfidfVectorizer
    col_fields: np.ndarray       # which field (g/k/c/d) each column belongs to
    tfidf_matrix: sparse.csr_matrix  # one weighted row per movie (length <= 1)
    presence: sparse.csc_matrix  # 1 where a movie has a term, else 0 (for query_scores)
    sentiment: np.ndarray        # VADER score per movie, -1..1
    has_overview: np.ndarray     # False where sentiment is just a neutral placeholder
    cert_bucket: list            # 'family' / 'teen' / 'adult' / 'unrated'


def build_features(movies):
    """Compute all features from scratch (about a second for 1000 movies)."""
    vectorizer = TfidfVectorizer(
        lowercase=False,        # _movie_text already lowercased everything
        token_pattern=r"\S+",   # tokens are separated by spaces, nothing fancier
        sublinear_tf=True,      # a word repeated 3x counts a bit more, not 3x more
        norm=None,              # we do our own normalising per field
    )
    raw = vectorizer.fit_transform([_movie_text(m) for m in movies])
    col_fields = np.array([t.split("_", 1)[0] for t in vectorizer.get_feature_names_out()])

    overviews = [(m.get("overview") or "").strip() for m in movies]
    tfidf_matrix = _apply_field_weights(raw, col_fields)
    return NLPFeatures(
        movie_ids=[m.get("id") for m in movies],
        vectorizer=vectorizer,
        col_fields=col_fields,
        tfidf_matrix=tfidf_matrix,
        presence=(tfidf_matrix > 0).astype(np.float32).tocsc(),
        sentiment=np.array([sentiment_score(o) for o in overviews]),
        has_overview=np.array([bool(o) for o in overviews]),
        cert_bucket=[certification_bucket(m.get("certification")) for m in movies],
    )


@st.cache_resource(show_spinner=False)
def _cached_features(_movies, movie_ids, weights):
    # Streamlit re-runs the whole script on every click. cache_resource keeps
    # the result in memory so it is computed once, not on every click.
    # `_movies` (leading underscore) is skipped by Streamlit's hashing; the cheap
    # `movie_ids` and `weights` are what decide whether the cache is still valid.
    return build_features(_movies)


def get_features(movies):
    """Return the (cached) NLPFeatures for this list of movies.

    Recomputed when the set of movies or the field weights change, but not
    when the text of the same movies changes.
    """
    return _cached_features(
        movies, tuple(m.get("id") for m in movies), tuple(FIELD_WEIGHTS.items())
    )


def similarity_matrix(features):
    """Movie-to-movie similarity as a dense (n x n) array, 0 to 1.

    Entry [i][j] is the weighted blend of genre/keyword/cast/director
    similarity described in FIELD_WEIGHTS. A movie missing a field can't
    reach 1.0 even against itself, because it has nothing to agree on there.
    """
    return linear_kernel(features.tfidf_matrix)
