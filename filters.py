"""The filtering logic.

This module narrows down the list of movies based on what the user asks
for. It removes movies that don't fit *before* they are ranked: the ids it
returns are passed to recommender.recommend(candidate_ids=...).

So far it only filters by runtime. By default it drops anything shorter than
60 minutes, because the cache contains animated shorts and TV specials
(30 of the 1000 movies) that otherwise rank above real feature films.
"""

# Anything shorter than this is treated as a short or TV special. The UI can
# turn the rule off by passing min_runtime=None.
DEFAULT_MIN_RUNTIME = 60


def apply_filters(movies, min_runtime=DEFAULT_MIN_RUNTIME, max_runtime=None):
    """Return the ids of the movies that pass every filter.

    min_runtime: shortest allowed runtime in minutes (inclusive). Defaults to
        DEFAULT_MIN_RUNTIME; pass None to turn the rule off.
    max_runtime: longest allowed runtime in minutes (inclusive), or None for
        no limit. "Under 2 hours" is max_runtime=119.

    Movies whose runtime is unknown (missing in TMDB) are kept: a filter only
    removes a movie when it can show that it doesn't fit.
    """
    ids = []
    for movie in movies:
        runtime = movie.get("runtime")
        if runtime is not None:
            if min_runtime is not None and runtime < min_runtime:
                continue
            if max_runtime is not None and runtime > max_runtime:
                continue
        ids.append(movie["id"])
    return ids
