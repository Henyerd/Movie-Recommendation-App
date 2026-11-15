"""Scoring and ranking.

This module decides which movies to recommend. Every movie gets three
scores, each on a 0-to-1 scale, which are blended into one final score:

  content   - how well the movie matches what the user asked for (a typed
              description, a movie they picked, or both). From nlp.py.
  sentiment - how close the movie's tone is to the tone the user wants.
  quality   - a fairness-adjusted star rating (see weighted_ratings below).

The final score is the weighted sum of the three. Every result carries all
three component scores, so the UI can say *why* a movie was recommended.

An optional fourth score, recency, can be switched on to nudge results toward
newer films (or toward classics). It is off by default.

Only films that actually match the search are returned. If the user typed a
query we recognise, a film with a content score of 0 is never shown, however
well rated it is. So if only 16 films match, you get 16 (not 16 plus filler).

Main entry points:
    recommend(...)             - ranked list for a query and/or a seed movie
    recommend_with_count(...)  - the same, plus how many films matched in total
    similar_movies(id)         - "more like this": content similarity alone
"""

import numpy as np

import nlp
import tmdb_client

# ---------------------------------------------------------------------------
# Settings - change the blend here. Only the ratios matter (they are
# rescaled to add up to 1), but keeping them at 100% total is easiest to read.
# ---------------------------------------------------------------------------
WEIGHTS = {"content": 0.55, "sentiment": 0.25, "quality": 0.20}

# Quality formula strength: a movie is treated as if it already had this many
# extra votes at the corpus-average rating. 500 is roughly the 25th percentile
# of vote counts in our 1000 movies (median is ~2100).
QUALITY_PRIOR_VOTES = 500

DEFAULT_N = 10

# Recency preference (the "newer / classics" slider). A soft nudge: when it is
# on, recency joins the blend as a fourth component. At full strength it gets
# this much weight (before everything is rescaled to add up to 1); half
# strength gets half of it. Off (0) means the component is left out entirely,
# so results are exactly as if the feature didn't exist.
RECENCY_MAX_WEIGHT = 0.30

# How fast "newness" fades. The newest film in the corpus scores 1, a film this
# many years older scores 0, and everything in between is a straight line.
RECENCY_SPAN_YEARS = 40


# ---------------------------------------------------------------------------
# Component 3: quality
# ---------------------------------------------------------------------------
def weighted_ratings(movies, prior_votes=QUALITY_PRIOR_VOTES):
    """Fairness-adjusted rating for every movie (still on TMDB's 0-10 scale).

    A raw average can't be trusted when few people voted: three fans giving
    10/10 would beat a classic rated 8.2 by 50,000 people. The fix is the
    "Bayesian average" (the same idea IMDb uses for its Top 250):

        WR = (v / (v + m)) * R  +  (m / (v + m)) * C

        R = the movie's own average rating      v = its number of votes
        C = the average rating of all movies    m = QUALITY_PRIOR_VOTES

    In words: blend the movie's own rating with the corpus average, and let
    the number of votes decide the mix. With few votes (v much smaller than
    m) the result stays close to C; with many votes (v much larger than m)
    it is essentially R. Example with m=500, C=7.6: a 9.0 from 80 votes
    becomes ~7.8, while an 8.2 from 50,000 votes stays ~8.2. Movies with zero
    votes get exactly C: no evidence, so no bonus and no penalty.
    """
    votes = np.array([m.get("vote_count") or 0 for m in movies], dtype=float)
    averages = np.array([m.get("vote_average") or 0 for m in movies], dtype=float)
    corpus_mean = averages[votes > 0].mean() if (votes > 0).any() else 0.0
    trust = votes / (votes + prior_votes)  # 0 = ignore own rating, ->1 = trust it
    return trust * averages + (1 - trust) * corpus_mean


def quality_scores(movies, prior_votes=QUALITY_PRIOR_VOTES):
    """Weighted ratings stretched to 0-1 across the corpus (worst=0, best=1).

    Without this stretch, ratings only span about 7 to 8.7 out of 10 in our
    corpus, so the quality weight would have far less real influence than the
    20% it is labelled with. Stretching puts quality on the same footing as
    the other components.
    """
    wr = weighted_ratings(movies, prior_votes)
    spread = wr.max() - wr.min()
    return (wr - wr.min()) / spread if spread > 0 else np.full(len(wr), 0.5)


# ---------------------------------------------------------------------------
# Optional component 4: recency
# ---------------------------------------------------------------------------
def recency_scores(movies, span=RECENCY_SPAN_YEARS):
    """How new each film is, from 0 (span years older than the newest film) to 1.

        score = 1 - (newest year in corpus - film's year) / span, floored at 0

    "Newest" is taken from the data, not today's date, so scores don't drift
    as the calendar moves. A film with no release year gets 0.5 (neutral).
    """
    years = np.array([m.get("release_year") or np.nan for m in movies], dtype=float)
    score = np.clip(1.0 - (np.nanmax(years) - years) / span, 0.0, 1.0)
    return np.where(np.isnan(years), 0.5, score)


# ---------------------------------------------------------------------------
# Component 2: sentiment (tone) match
# ---------------------------------------------------------------------------
def sentiment_match(tone, movie_tones):
    """How close each movie's tone is to the wanted tone, from 0 to 1.

    `tone` runs from -1 (dark) to +1 (light), the same scale as the VADER
    scores in nlp.py. Distance can be at most 2, so:
        match = 1 - |wanted tone - movie tone| / 2
    A perfect tone match gives 1.0; the opposite extreme gives 0.0.
    """
    tone = float(np.clip(tone, -1.0, 1.0))
    return 1.0 - np.abs(movie_tones - tone) / 2.0


# ---------------------------------------------------------------------------
# Component 1: content similarity
# ---------------------------------------------------------------------------
def query_has_matches(query, movies=None, features=None, tone=None):
    """True if at least one word of the query is something we can match.

    Lets the UI warn "we didn't understand that" instead of quietly showing
    a list that ignores what was typed. Pass the same `tone` you give to
    recommend(): with a tone, mood words are ignored, so a query made only of
    mood words ("something dark and gritty") returns False. That isn't an
    error; those results are then ranked by tone and quality alone.
    """
    movies, features = _resolve(movies, features)
    return nlp.query_scores(query, features, strip_mood=tone is not None) is not None


def _similarity_to_seed(features, index):
    """Content similarity of every movie to the movie at row `index`.

    Uses cosine similarity of the TF-IDF vectors: with a whole movie as the
    input, "same overall shape" is exactly the right question.
    """
    matrix = features.tfidf_matrix
    return (matrix @ matrix[index].T).toarray().ravel()


def _similarity_to_query(features, query, strip_mood=False):
    """Content score of every movie for a typed query: the share of the
    query's words the movie matches (see nlp.query_scores). None if the query
    contains no words we recognise."""
    return nlp.query_scores(query, features, strip_mood=strip_mood)


# ---------------------------------------------------------------------------
# Plumbing
# ---------------------------------------------------------------------------
def _resolve(movies, features):
    """Fill in the movie list and NLP features if the caller didn't pass them.

    The app should pass its own already-loaded `movies` to avoid re-reading
    the cache file on every call.
    """
    if movies is None:
        movies = tmdb_client.load_movies()
    if features is None:
        features = nlp.get_features(movies)
    return movies, features


def _index_of(movies, movie_id):
    for i, movie in enumerate(movies):
        if movie.get("id") == movie_id:
            return i
    raise ValueError(f"No movie with id {movie_id} in the corpus.")


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------
def recommend(query=None, **kwargs):
    """Rank movies for a user and return the top `n` (see recommend_with_count)."""
    return recommend_with_count(query, **kwargs)[0]


def recommend_with_count(
    query=None,
    *,
    seed_id=None,
    tone=None,
    recency=0.0,
    n=DEFAULT_N,
    candidate_ids=None,
    exclude_ids=(),
    movies=None,
    features=None,
):
    """Rank movies for a user. Returns (top `n` results, total films that matched).

    The total lets the UI say "only 16 films match" when fewer than `n` do.
    With no usable query or seed, every eligible film counts as a match.

    Inputs (all optional, use any combination):
        query          free text, e.g. "something funny"
        seed_id        TMDB id of a movie the user likes ("like this one")
        tone           wanted tone from -1 (dark) to +1 (light)
        recency        -1 (favour classics) .. 0 (no preference, the default) ..
                       +1 (favour new films). A nudge, never a filter.
        candidate_ids  only rank these movie ids. This is where the output of
                       filters.py plugs in (runtime, genre, certification...)
        exclude_ids    ids never to return, e.g. movies already seen

    For a typed query the content score is the share of the query's words the
    movie matches (mood words like "dark" are ignored when `tone` is given,
    so tone isn't counted twice); for a seed movie it is TF-IDF cosine
    similarity. A film that matches none of the query/seed (content 0) is not
    shown at all. If both
    are given the two are averaged (the query score tends to run higher, so it
    will dominate a little). A component with no input is left out and the remaining weights
    are rescaled to add up to 1: no tone -> sentiment is skipped; no
    recognisable query/seed -> content is skipped. (Giving every movie the
    same score on a component wouldn't change the order anyway.)

    Each result is a dict:
        movie          the movie's full dict from the cache
        score          final score, 0 to 1
        components     {"content", "sentiment", "quality", "recency"} each 0 to 1,
                       or None for a component that was skipped
        weights        the weights actually used (sum to 1)
        contributions  component * weight; these add up to `score`
        details        weighted_rating, movie_tone, and tone_estimated
                       (True when the movie has no overview, see below)

    Movies with no overview have no tone to measure. Instead of a penalty or
    a bonus they get the *average* tone-match of the other candidates for
    this request, so they land exactly in the middle of the pack on that
    component.
    """
    movies, features = _resolve(movies, features)
    size = len(movies)
    ids = [m.get("id") for m in movies]

    # --- which movies are eligible -----------------------------------------
    eligible = np.ones(size, dtype=bool)
    if candidate_ids is not None:
        eligible &= np.isin(ids, list(candidate_ids))
    excluded = set(exclude_ids)
    if seed_id is not None:
        seed_index = _index_of(movies, seed_id)
        excluded.add(seed_id)
    eligible &= ~np.isin(ids, list(excluded))
    if not eligible.any():
        return [], 0

    # --- component 1: content ----------------------------------------------
    similarities = []
    if query:
        # Tone is scored separately, so mood words ("dark", "light") are
        # dropped from the query when a tone is given. Without a tone they stay.
        from_query = _similarity_to_query(features, query, strip_mood=tone is not None)
        if from_query is not None:
            similarities.append(from_query)
    if seed_id is not None:
        similarities.append(_similarity_to_seed(features, seed_index))
    content = np.mean(similarities, axis=0) if similarities else None
    if content is not None:
        eligible &= content > 0       # no padding: unrelated films are never shown
        if not eligible.any():
            return [], 0
    total = int(eligible.sum())

    # --- component 2: sentiment --------------------------------------------
    sentiment = None
    if tone is not None:
        sentiment = sentiment_match(tone, features.sentiment)
        measurable = eligible & features.has_overview
        neutral = sentiment[measurable].mean() if measurable.any() else 0.5
        sentiment = np.where(features.has_overview, sentiment, neutral)

    # --- component 3: quality ----------------------------------------------
    quality = quality_scores(movies)

    # --- blend ---------------------------------------------------------------
    newness = None
    if recency:
        newness = recency_scores(movies)
        if recency < 0:               # favouring classics: flip the scale
            newness = 1.0 - newness
    parts = {"content": content, "sentiment": sentiment, "quality": quality, "recency": newness}
    base = dict(WEIGHTS, recency=RECENCY_MAX_WEIGHT * abs(recency))
    active = {k: base[k] for k, v in parts.items() if v is not None}
    total_weight = sum(active.values())
    weights = {k: w / total_weight for k, w in active.items()}
    score = sum(weights[k] * parts[k] for k in weights)

    pool = np.flatnonzero(eligible)
    best = pool[np.argsort(-score[pool], kind="stable")][:n]

    wr = weighted_ratings(movies)
    results = []
    for i in best:
        components = {k: (float(v[i]) if v is not None else None) for k, v in parts.items()}
        results.append(
            {
                "movie": movies[i],
                "score": float(score[i]),
                "components": components,
                "weights": weights,
                "contributions": {k: weights[k] * components[k] for k in weights},
                "details": {
                    "weighted_rating": float(wr[i]),
                    "movie_tone": float(features.sentiment[i]) if features.has_overview[i] else None,
                    "tone_estimated": bool(tone is not None and not features.has_overview[i]),
                },
            }
        )
    return results, total


def similar_movies(movie_id, n=DEFAULT_N, movies=None, features=None):
    """"More like this": the movies most similar to `movie_id`, by content only.

    Ignores tone and quality completely. Returns a list of
    {"movie": ..., "similarity": 0-to-1}, best first, excluding the movie
    itself.
    """
    movies, features = _resolve(movies, features)
    index = _index_of(movies, movie_id)
    similarity = _similarity_to_seed(features, index)
    similarity[index] = -1  # never return the movie itself
    best = [i for i in np.argsort(-similarity, kind="stable")[:n] if similarity[i] > 0]  # no zero-similarity padding
    return [{"movie": movies[i], "similarity": float(similarity[i])} for i in best]
