"""Tiebreaker: a movie recommender for couples deciding what to watch tonight.

This is the file you run (streamlit run app.py). Streamlit turns Python code
into a web page, and the page is rebuilt from the top every time the user
touches a widget. So the shape of this file is always the same:

    1. load the movie data (once, cached)
    2. read the sidebar widgets -> a dict of filter settings
    3. filters.apply_filters()   -> ids of the movies allowed through
    4. recommender.recommend()   -> ranked list, using ONLY those ids
    5. draw the results as cards, each explaining why it was picked

Step 3 feeding step 4 (candidate_ids) is what makes the sidebar filters work.
"""

import html

import streamlit as st

import filters
import nlp
import recommender
import tmdb_client

st.set_page_config(page_title="Tiebreaker", page_icon="🎬", layout="wide")

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
POSTER_URL = "https://image.tmdb.org/t/p/w342"   # TMDB's image CDN + poster_path
RESULTS = recommender.DEFAULT_N                   # cards per page (10)
MIN_PLATFORM_MOVIES = 10   # platforms with fewer movies than this are left out of the dropdown
RUNTIME_MAX = 240          # right end of the runtime slider (= "no maximum")

# Colours used for the three "why" bars (the page theme is in .streamlit/config.toml)
COLOURS = {"content": "#F5B841", "sentiment": "#8E7CFF", "quality": "#4CC38A"}
ROLES = {"Anyone": "any", "Actors": "actor", "Directors": "director"}
MOOD_STEPS = [round(i * 0.2, 1) for i in range(-5, 6)]  # -1.0 (dark) ... +1.0 (light)

STYLE = """
<style>
.block-container { padding-top: 2.2rem; max-width: 1200px; }
.tb-brand { display: flex; align-items: center; gap: 14px; margin-bottom: 4px; }
.tb-wordmark { font-size: 2.7rem; font-weight: 800; letter-spacing: -0.04em; line-height: 1; }
.tb-wordmark span { color: #F5B841; }
.tb-tagline { color: #9AA0AE; font-size: 1.02rem; margin: 6px 0 16px 0; }
.tb-title { font-size: 1.35rem; font-weight: 700; line-height: 1.25; }
.tb-meta { color: #9AA0AE; font-size: 0.93rem; margin-top: 2px; }
.tb-head { display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; }
.tb-score { text-align: center; min-width: 58px; padding: 6px 10px; border-radius: 12px;
            background: rgba(245,184,65,.14); color: #F5B841; }
.tb-score b { display: block; font-size: 1.5rem; line-height: 1.1; }
.tb-score small { font-size: 0.68rem; letter-spacing: .08em; text-transform: uppercase; }
.tb-chips { margin: 10px 0 2px 0; }
.tb-chip { display: inline-block; padding: 2px 11px; margin: 0 6px 6px 0; border-radius: 999px;
           background: rgba(255,255,255,.08); font-size: 0.8rem; }
.tb-chip.stream { background: rgba(245,184,65,.16); color: #F5B841; }
.tb-chip.none { background: transparent; border: 1px dashed rgba(255,255,255,.22); color: #9AA0AE; }
.tb-plot { color: #C3C7D2; font-size: 0.9rem; margin: 6px 0 10px 0; line-height: 1.45; }
.tb-why { border-top: 1px solid rgba(255,255,255,.09); padding-top: 10px; margin-top: 4px; }
.tb-why-title { font-size: 0.72rem; letter-spacing: .09em; text-transform: uppercase; color: #9AA0AE; margin-bottom: 6px; }
.tb-why-row { display: grid; grid-template-columns: 112px 1fr 42px; align-items: center; gap: 10px; margin-top: 7px; }
.tb-why-label { font-size: 0.85rem; font-weight: 600; }
.tb-bar { height: 8px; border-radius: 4px; background: rgba(255,255,255,.1); overflow: hidden; }
.tb-bar div { height: 100%; border-radius: 4px; }
.tb-pts { text-align: right; font-size: 0.85rem; color: #C3C7D2; font-variant-numeric: tabular-nums; }
.tb-why-text { grid-column: 2 / 4; color: #9AA0AE; font-size: 0.82rem; margin-top: -3px; line-height: 1.4; }
.tb-noposter { aspect-ratio: 2 / 3; border-radius: 10px; border: 1px dashed rgba(255,255,255,.2);
               display: flex; align-items: center; justify-content: center; color: #9AA0AE; font-size: .85rem; }
div[data-testid="stImage"] img { border-radius: 10px; max-width: 220px !important; }
.tb-noposter { max-width: 220px; }
</style>
"""

# Two overlapping rings: two people, one shared choice.
LOGO = (
    '<svg width="44" height="44" viewBox="0 0 44 44" fill="none" aria-hidden="true">'
    '<circle cx="17" cy="22" r="13" stroke="#F5B841" stroke-width="4"/>'
    '<circle cx="27" cy="22" r="13" stroke="#EDEEF2" stroke-width="4"/></svg>'
)


# ---------------------------------------------------------------------------
# Cached work. Streamlit re-runs this whole file on every click, so anything
# slow is wrapped in a cache. st.cache_resource keeps ONE shared object (the
# loaded movies); st.cache_data remembers the RESULT for a given set of inputs.
# Arguments starting with an underscore are not hashed (too big to hash).
# `size` is only there so the caches refresh if the movie list changes.
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading movies...")
def load_catalogue():
    movies = tmdb_client.load_movies()
    return movies, nlp.get_features(movies)


@st.cache_data(show_spinner=False)
def filter_options(_movies, size):
    """Everything the sidebar dropdowns need, with movie counts."""
    years = [m["release_year"] for m in _movies if m.get("release_year")]
    return {
        "platforms": [p for p in filters.available_platforms(_movies) if p[2] >= MIN_PLATFORM_MOVIES],
        "genres": filters.available_genres(_movies),
        "languages": filters.available_languages(_movies),
        "decades": filters.available_decades(_movies),
        "certifications": filters.available_certifications(_movies),
        "year_min": min(years),
        "year_max": max(years),
    }


@st.cache_data(show_spinner=False)
def run_filters(_movies, size, criteria):
    """Which movies pass the filters, and a plain-language explanation."""
    ids = filters.apply_filters(_movies, **criteria)
    return ids, filters.explain_filters(_movies, **criteria)


@st.cache_data(show_spinner=False, max_entries=128)
def run_recommend(_movies, _features, size, query, tone, candidate_ids, n):
    return recommender.recommend(
        query or None, tone=tone, n=n, candidate_ids=candidate_ids,
        movies=_movies, features=_features,
    )


@st.cache_data(show_spinner=False, max_entries=64)
def run_similar(_movies, _features, size, movie_id, candidate_ids, n):
    """'More like this', content similarity only, limited to movies that pass the filters."""
    allowed = set(candidate_ids)
    ranked = recommender.similar_movies(movie_id, n=len(_movies), movies=_movies, features=_features)
    return [r for r in ranked if r["movie"]["id"] in allowed][:n]


# ---------------------------------------------------------------------------
# Small text helpers
# ---------------------------------------------------------------------------
def esc(value):
    """Make text safe to place inside HTML."""
    return html.escape(str(value))


def tone_label(tone):
    """A number from -1 to +1 as a word."""
    if tone <= -0.6:
        return "very dark"
    if tone <= -0.2:
        return "dark"
    if tone < 0.2:
        return "balanced"
    if tone < 0.6:
        return "light"
    return "very light"


def runtime_text(minutes):
    if not minutes:
        return "runtime unknown"
    hours, mins = divmod(minutes, 60)
    return f"{hours}h {mins:02d}m" if hours else f"{mins}m"


def shorten(text, limit=210):
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "..."


# ---------------------------------------------------------------------------
# Sidebar: every widget stores its value in st.session_state under a key, so
# the "Reset filters" button can put them all back to their defaults.
# ---------------------------------------------------------------------------
def default_values(options):
    return {
        "mood_on": False, "mood": 0.0,
        "platforms": [],
        "genres_all": [], "genres_any": [], "genres_none": [],
        "years": (options["year_min"], options["year_max"]), "decades": [],
        "min_rating": 0.0, "min_votes": 0,
        "hide_shorts": True, "runtime": (0, RUNTIME_MAX),
        "certifications": [], "languages": [],
        "person": "", "person_role": "Anyone",
    }


def reset_filters(defaults):
    for key, value in defaults.items():
        st.session_state[key] = value
    st.session_state["more_like"] = None


def labels_with_counts(rows):
    """[(value, label, count)] -> {value: "label (count)"} for dropdown display."""
    return {value: f"{label} ({count})" for value, label, count in rows}


def render_sidebar(options, defaults):
    """Draw the filters. Returns (criteria for filters.py, wanted tone or None)."""
    ss = st.session_state
    st.sidebar.markdown("### Filters")
    st.sidebar.button("Reset filters", on_click=reset_filters, args=(defaults,), width="stretch")

    with st.sidebar.expander("Mood", expanded=True):
        st.toggle("Match a mood", key="mood_on",
                  help="Prefer movies whose plot reads darker or lighter. Mood words in your search "
                       "(dark, light, gritty...) are then ignored so the mood isn't counted twice.")
        if ss["mood_on"]:
            st.select_slider("Dark to light", options=MOOD_STEPS, key="mood",
                             format_func=lambda v: f"{tone_label(v).capitalize()} ({v:+.1f})")

    with st.sidebar.expander("Where to watch", expanded=True):
        names = labels_with_counts(options["platforms"])
        st.multiselect("Streaming in Canada", list(names), key="platforms", format_func=names.get,
                       placeholder="Any service",
                       help=f"Only services with at least {MIN_PLATFORM_MOVIES} movies are listed. "
                            "Subscription streaming only; rent and buy are not counted.")

    with st.sidebar.expander("Genres"):
        names = labels_with_counts(options["genres"])
        st.multiselect("Must have all of", list(names), key="genres_all", format_func=names.get,
                       placeholder="Any", help="AND: every genre chosen must be present.")
        st.multiselect("At least one of", list(names), key="genres_any", format_func=names.get,
                       placeholder="Any", help="OR: any one of the genres chosen is enough.")
        st.multiselect("Exclude", list(names), key="genres_none", format_func=names.get,
                       placeholder="None", help="NOT: movies with any of these genres are removed.")

    with st.sidebar.expander("Release date"):
        st.slider("Release year", options["year_min"], options["year_max"], key="years")
        names = labels_with_counts(options["decades"])
        st.multiselect("Decade", list(names), key="decades", format_func=names.get, placeholder="Any")

    with st.sidebar.expander("Quality"):
        st.slider("Minimum rating", 0.0, 9.0, step=0.5, key="min_rating",
                  help="TMDB's average user rating out of 10. 0 means no minimum.")
        st.select_slider("Minimum votes", options=[0, 50, 100, 500, 1000, 5000, 10000, 20000], key="min_votes",
                         help="Ignore movies rated by only a handful of people. 0 means no minimum.")

    with st.sidebar.expander("Length, rating and language"):
        st.toggle(f"Hide shorts and TV specials (under {filters.DEFAULT_MIN_RUNTIME} min)", key="hide_shorts")
        st.slider("Runtime (minutes)", 0, RUNTIME_MAX, step=5, key="runtime",
                  help=f"Drag the right end all the way to {RUNTIME_MAX} for no maximum.")
        names = labels_with_counts(options["certifications"])
        st.multiselect("Age rating", list(names), key="certifications", format_func=names.get,
                       placeholder="Any",
                       help="Family: G, PG. Teen: PG-13, 14A. Adult: R, 18A. "
                            "Unrated: no rating, or marked NR.")
        names = labels_with_counts(options["languages"])
        st.multiselect("Original language", list(names), key="languages", format_func=names.get,
                       placeholder="Any")

    with st.sidebar.expander("People"):
        st.text_input("Actor or director", key="person", placeholder="e.g. Nolan",
                      help="Part of a name is enough. Only the top 5 billed actors are searched.")
        st.radio("Look for them as", list(ROLES), key="person_role", horizontal=True)

    # --- turn the widget values into the settings filters.py understands ---
    year_low, year_high = ss["years"]
    runtime_low, runtime_high = ss["runtime"]
    min_runtime = runtime_low or None
    if ss["hide_shorts"]:
        min_runtime = max(min_runtime or 0, filters.DEFAULT_MIN_RUNTIME)
    criteria = {
        "year_from": year_low if year_low > options["year_min"] else None,
        "year_to": year_high if year_high < options["year_max"] else None,
        "decades": list(ss["decades"]),
        "min_rating": ss["min_rating"] or None,
        "min_votes": ss["min_votes"] or None,
        "min_runtime": min_runtime,
        "max_runtime": runtime_high if runtime_high < RUNTIME_MAX else None,
        "certifications": list(ss["certifications"]),
        "languages": list(ss["languages"]),
        "person": ss["person"].strip() or None,
        "person_role": ROLES[ss["person_role"]],
        "genres_all": list(ss["genres_all"]),
        "genres_any": list(ss["genres_any"]),
        "genres_none": list(ss["genres_none"]),
        "platforms": list(ss["platforms"]),
    }
    tone = ss["mood"] if ss["mood_on"] else None
    return criteria, tone


def count_active_filters(criteria, options):
    """How many filters the user has set (the default shorts rule doesn't count)."""
    groups = [
        criteria["year_from"] is not None or criteria["year_to"] is not None,
        bool(criteria["decades"]), criteria["min_rating"] is not None,
        criteria["min_votes"] is not None,
        criteria["max_runtime"] is not None or criteria["min_runtime"] not in (None, filters.DEFAULT_MIN_RUNTIME),
        bool(criteria["certifications"]), bool(criteria["languages"]),
        criteria["person"] is not None, bool(criteria["genres_all"]),
        bool(criteria["genres_any"]), bool(criteria["genres_none"]), bool(criteria["platforms"]),
    ]
    return sum(groups)


# ---------------------------------------------------------------------------
# Result cards
# ---------------------------------------------------------------------------
def why_rows(result, query, tone):
    """The "why this pick" explanation, built from recommender.py's component scores.

    Returns one dict per component that was actually used: its colour, label,
    0-1 score, points contributed to the final score, and a sentence.
    """
    parts, details = result["components"], result["details"]
    rows = []

    if parts["content"] is not None:
        pct = round(parts["content"] * 100)
        text = ("Matches everything you searched for." if pct >= 100
                else f"Matches {pct}% of what you searched for.")
        rows.append(("content", "Search match", parts["content"], text))

    if parts["sentiment"] is not None:
        if details["tone_estimated"]:
            text = "No plot summary to read, so its mood counts as average."
        else:
            text = (f"You wanted {tone_label(tone)}; its plot reads "
                    f"{tone_label(details['movie_tone'])} ({details['movie_tone']:+.2f}).")
        rows.append(("sentiment", "Mood fit", parts["sentiment"], text))

    votes = result["movie"].get("vote_count") or 0
    text = f"Weighted rating {details['weighted_rating']:.1f}/10 ({votes:,} votes)."
    if votes < recommender.QUALITY_PRIOR_VOTES:
        text += " Few votes, so the score is pulled toward the average."
    rows.append(("quality", "Quality", parts["quality"], text))

    return [
        {"colour": COLOURS[k], "label": label, "score": s, "points": result["contributions"][k] * 100, "text": text}
        for k, label, s, text in rows
    ]


def shared_traits(seed, movie):
    """What a 'more like this' result has in common with the movie it is like."""
    found = []
    genres = [g for g in movie.get("genres", []) if g in seed.get("genres", [])]
    if genres:
        found.append("Genres: " + ", ".join(genres))
    keywords = [k for k in movie.get("keywords", []) if k in seed.get("keywords", [])][:4]
    if keywords:
        found.append("Themes: " + ", ".join(keywords))
    directors = set(filter(None, (seed.get("director") or "").split(", "))) & set(
        filter(None, (movie.get("director") or "").split(", ")))
    if directors:
        found.append("Director: " + ", ".join(sorted(directors)))
    cast = [c for c in movie.get("cast", []) if c in seed.get("cast", [])]
    if cast:
        found.append("Cast: " + ", ".join(cast))
    return found


def similar_rows(result, seed):
    traits = shared_traits(seed, result["movie"])
    text = " · ".join(traits) if traits else "Similar overall mix of genres, themes and credits."
    return [{"colour": COLOURS["content"], "label": "Similar content", "score": result["similarity"],
             "points": result["similarity"] * 100, "text": text}]


def render_why(rows, show_points=True):
    body = ""
    for row in rows:
        points = f"+{row['points']:.0f}" if show_points else f"{row['score'] * 100:.0f}%"
        body += (
            f'<div class="tb-why-row"><div class="tb-why-label">{esc(row["label"])}</div>'
            f'<div class="tb-bar"><div style="width:{max(row["score"], 0) * 100:.0f}%;background:{row["colour"]}"></div></div>'
            f'<div class="tb-pts">{points}</div>'
            f'<div class="tb-why-text">{esc(row["text"])}</div></div>'
        )
    return f'<div class="tb-why"><div class="tb-why-title">Why this pick</div>{body}</div>'


def render_card(movie, score, why_html, key, score_label="match"):
    """One result: poster on the left, details and the explanation on the right."""
    with st.container(border=True):
        poster, info = st.columns([1, 3.2], gap="medium")
        with poster:
            if movie.get("poster_path"):
                st.image(POSTER_URL + movie["poster_path"], width="stretch")
            else:
                st.markdown('<div class="tb-noposter">No poster</div>', unsafe_allow_html=True)
        with info:
            rating = movie.get("vote_average") or 0
            votes = movie.get("vote_count") or 0
            meta = [str(movie.get("release_year") or "year unknown"), runtime_text(movie.get("runtime"))]
            if movie.get("certification", "").strip():
                meta.append(movie["certification"].strip())
            meta.append(f"★ {rating:.1f} ({votes:,})" if votes else "no ratings yet")

            genres = "".join(f'<span class="tb-chip">{esc(g)}</span>' for g in movie.get("genres", []))
            platforms = filters.streaming_platforms(movie)
            streams = ("".join(f'<span class="tb-chip stream">{esc(p)}</span>' for p in platforms)
                       if platforms else '<span class="tb-chip none">Not streaming in Canada</span>')

            st.markdown(
                f'<div class="tb-head"><div><div class="tb-title">{esc(movie["title"])}</div>'
                f'<div class="tb-meta">{esc(" · ".join(meta))}</div></div>'
                f'<div class="tb-score"><b>{round(score * 100)}</b><small>{esc(score_label)}</small></div></div>'
                f'<div class="tb-chips">{genres}</div>'
                f'<div class="tb-chips"><span class="tb-meta">Streaming: </span>{streams}</div>'
                f'<div class="tb-plot">{esc(shorten(movie.get("overview")))}</div>'
                f'{why_html}',
                unsafe_allow_html=True,
            )
            st.button("More like this", key=key, on_click=open_more_like, args=(movie["id"],))


def open_more_like(movie_id):
    st.session_state["more_like"] = movie_id
    st.session_state["scroll_top"] = True  # the new list should start at its heading


def close_more_like():
    st.session_state["more_like"] = None


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------
def main():
    st.markdown(STYLE, unsafe_allow_html=True)

    if not tmdb_client.CACHE_PATH.exists():
        st.error("No movie data yet. In a terminal, run `python build_cache.py` (about 4 minutes), "
                 "then reload this page.")
        st.stop()

    movies, features = load_catalogue()
    size = len(movies)
    options = filter_options(movies, size)
    defaults = default_values(options)
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    st.session_state.setdefault("more_like", None)
    by_id = {m["id"]: m for m in movies}

    criteria, tone = render_sidebar(options, defaults)

    # --- header -------------------------------------------------------------
    st.markdown(
        f'<div class="tb-brand">{LOGO}<div><div class="tb-wordmark">Tie<span>breaker</span></div></div></div>'
        '<div class="tb-tagline">Two people, one couch, one pick. Tell us the vibe and we will settle it.</div>',
        unsafe_allow_html=True,
    )
    query = st.text_input(
        "What are you in the mood for?", key="query", on_change=close_more_like,
        placeholder="e.g. something funny and light, a tense heist, animated for kids...",
        label_visibility="collapsed",
    ).strip()

    # --- step 3: filters -> the ids that are allowed through ------------------
    candidate_ids, explanation = run_filters(movies, size, criteria)
    candidate_ids = tuple(candidate_ids)
    active = count_active_filters(criteria, options)
    st.caption(f"{explanation['count']:,} of {size:,} movies match your filters"
               + (f" ({active} filter{'s' if active != 1 else ''} on)" if active else ""))

    # Fewer than 10 matches: say which filter to relax (the message from explain_filters)
    if explanation["message"]:
        st.warning(explanation["message"])
    if not candidate_ids:
        st.button("Reset all filters", on_click=reset_filters, args=(defaults,), type="primary")
        return

    # --- step 4 + 5: recommend (or "more like this"), then draw the cards ------
    seed_id = st.session_state["more_like"]
    if seed_id is not None:
        if st.session_state.pop("scroll_top", False):
            # A "More like this" button may have been clicked far down the page;
            # jump back up so the user sees the new heading, not the middle of the list.
            st.html("<script>setTimeout(() => { const m = document.querySelector('[data-testid=\"stMain\"]');"
                    " if (m) m.scrollTo({top: 0}); }, 60);</script>", unsafe_allow_javascript=True)
        seed = by_id[seed_id]
        left, right = st.columns([4, 1])
        left.subheader(f"More like {seed['title']}" + (f" ({seed['release_year']})" if seed.get("release_year") else ""))
        right.button("← Back to picks", on_click=close_more_like, width="stretch")
        st.caption("Ranked by similar genres, themes, cast and director. Your filters still apply.")
        results = run_similar(movies, features, size, seed_id, candidate_ids, RESULTS)
        if not results:
            st.info("Nothing else like this passes your filters. Try relaxing one.")
        for rank, result in enumerate(results, 1):
            render_card(result["movie"], result["similarity"],
                        render_why(similar_rows(result, seed), show_points=False),
                        key=f"more_{rank}_{result['movie']['id']}", score_label="similar")
        return

    results = run_recommend(movies, features, size, query, tone, candidate_ids, RESULTS)

    if query:
        st.subheader(f"Top picks for “{query}”")
        if not recommender.query_has_matches(query, movies, features, tone):
            if tone is None:
                st.info("None of the words in your search matched a genre or theme, so these are ranked by "
                        "quality only. Try words like comedy, heist, animated or space.")
            else:
                st.info("None of the words in your search matched a genre or theme (mood words like dark or "
                        "light are handled by the mood slider), so these are ranked by mood and quality only.")
    else:
        st.subheader("Top picks tonight")
        st.caption("Type what you are in the mood for above to personalise the list. "
                   "Right now it ranks by quality" + (" and mood." if tone is not None else "."))

    # With no search and no mood there is nothing to "match", only quality to rank by
    label = "match" if (query or tone is not None) else "score"
    for rank, result in enumerate(results, 1):
        render_card(result["movie"], result["score"], render_why(why_rows(result, query, tone)),
                    key=f"more_{rank}_{result['movie']['id']}", score_label=label)


main()
