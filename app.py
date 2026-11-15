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

import base64
import html
from pathlib import Path

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
COLOURS = {"content": "#FF8A65", "sentiment": "#F27AA5", "quality": "#7BD9A5", "recency": "#FFD479"}
ROLES = {"Anyone": "any", "Actors": "actor", "Directors": "director"}
EXAMPLES = ["Funny and light", "A tense heist", "Animated for the whole family",
            "Date-night romance", "Mind-bending sci-fi"]   # one-tap searches in the hero
# The recency slider: -1 = favour classics ... 0 = no preference ... +1 = favour new films.
RECENCY_LABELS = {-1.0: "Mostly classics", -0.5: "Lean classic", 0.0: "No preference",
                  0.5: "Lean new", 1.0: "Mostly new"}
MOOD_STEPS = [round(i * 0.2, 1) for i in range(-5, 6)]  # -1.0 (dark) ... +1.0 (light)

STYLE = """
<style>
/* "Velvet Night": warm dark theme. Colours, fonts and base size come from
   .streamlit/config.toml; this block handles layout, cards and the details
   the theme file can't reach. Spacing follows an 8px grid (0.5rem steps). */
:root {
  --surface: #2B1D2E;  --surface-2: #35253A;
  --text: #F7EBDD;     --muted: #C9B6C4;
  --line: rgba(247,235,221,.14);
  --coral: #FF8A65;    --butter: #FFD479;
  --rose: #F27AA5;     /* second accent: everything to do with MOOD, plus the brand mark */
  --grad: linear-gradient(100deg, #F27AA5 0%, #FF8A65 42%, #FFD479 100%);
  /* film grain: SVG noise, white at very low alpha, tiled. Only ever drawn on the page
     background (behind every card and all text), never over content. */
  --grain: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='240' height='240'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.8' numOctaves='2' stitchTiles='stitch'/%3E%3CfeColorMatrix values='0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 .13 0 0 0 0'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
  /* the two-ring logo, drawn big for the hero background */
  --rings: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='2 7 40 30' fill='none' stroke-width='4'%3E%3Ccircle cx='17' cy='22' r='13' stroke='%23FF8A65'/%3E%3Ccircle cx='27' cy='22' r='13' stroke='%23F27AA5'/%3E%3C/svg%3E");
}

/* --- page background: two soft warm glows instead of a flat fill --------- */
.stApp {
  background:
    radial-gradient(ellipse 85% 75% at 50% 42%, transparent 55%, rgba(8,3,10,.6) 100%),   /* vignette: darker edges */
    var(--grain),
    radial-gradient(900px 520px at 90% -6%, rgba(255,122,89,.20), transparent 62%),
    radial-gradient(760px 480px at 4% 0%, rgba(255,212,121,.10), transparent 60%),
    radial-gradient(820px 560px at 12% 105%, rgba(242,122,165,.14), transparent 62%),
    #1E1420;
  background-attachment: fixed;
}
[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 1360px; padding: 2rem 2.5rem 4rem; }
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] { gap: 1.5rem; }
[data-testid="stCaptionContainer"] { font-size: 0.95rem; color: var(--muted); }
.stApp h3 { margin-bottom: 0; }
.st-key-results-head { gap: 0.25rem; }

/* --- hero ------------------------------------------------------------------ */
.st-key-hero {
  background: linear-gradient(135deg, rgba(255,122,89,.18), rgba(255,212,121,.07) 55%, rgba(43,29,46,.7));
  border: 1px solid var(--line); border-radius: 28px; padding: 2rem 2.5rem 2rem;
  box-shadow: 0 16px 48px rgba(0,0,0,.25); gap: 1rem;
  position: relative; overflow: hidden; isolation: isolate;
}
.st-key-hero::before {   /* the logo, oversized and faint, cropped by the hero's edge */
  content: ""; position: absolute; z-index: -1; pointer-events: none;
  right: -90px; top: 50%; width: 600px; height: 450px; transform: translateY(-50%) rotate(-8deg);
  background: var(--rings) no-repeat center / contain; opacity: .14;
}
.tb-brand { display: flex; align-items: center; gap: 16px; }
.tb-wordmark { font-family: "Fraunces", Georgia, serif; font-size: 3.5rem; font-weight: 700;
               letter-spacing: -0.02em; line-height: 1.1; padding-bottom: 4px;
               background: var(--grad); -webkit-background-clip: text; background-clip: text;
               -webkit-text-fill-color: transparent; color: transparent; }
.tb-tagline { color: #EBDCE6; font-size: 1.2rem; margin: 0; max-width: 40rem; }
.st-key-hero [data-testid="stTextInputRootElement"] { min-height: 3.5rem; border-radius: 16px; background: rgba(30,20,32,.75);
                                      border: 1px solid rgba(247,235,221,.25); }
.st-key-hero [data-baseweb="input"], .st-key-hero [data-baseweb="base-input"] { background: transparent; min-height: 3.5rem; }
.st-key-hero input { font-size: 1.1rem; padding: 0.75rem 1.25rem; min-height: 3.5rem; }
.st-key-hero [data-testid="stTextInputRootElement"]:focus-within { border-color: var(--coral); box-shadow: 0 0 0 3px rgba(255,138,101,.25); }
.tb-try { color: #E0CEDA; font-size: 0.95rem; margin: 0; }
/* example buttons: small pills that wrap onto several lines instead of stacking */
.st-key-hero [data-testid="stHorizontalBlock"] { flex-wrap: wrap; gap: 0.5rem; }
.st-key-hero [data-testid="stColumn"] { flex: 0 0 auto; width: auto; min-width: 0; }
.st-key-hero button { min-height: 2.25rem; padding: 0.25rem 1rem; font-size: 0.95rem; }

/* --- buttons ------------------------------------------------------------- */
button[data-testid="stBaseButton-secondary"] {
  border: 1.5px solid rgba(255,138,101,.6); border-radius: 999px; color: var(--text);
  background: transparent; font-weight: 600; padding: 0.25rem 1.25rem;
}
button[data-testid="stBaseButton-secondary"]:hover { background: rgba(255,138,101,.16); border-color: var(--coral); color: #fff; }
button[data-testid="stBaseButton-primary"] { border-radius: 999px; background: var(--coral); color: #2A1224; font-weight: 700; border: none; }

/* --- mood panel: rose accent (rose always means "mood") -------------------------------- */
[data-testid="stSidebar"] [data-testid="stExpander"]:has([class*="st-key-mood"]) {
  border-color: rgba(242,122,165,.5); box-shadow: inset 4px 0 0 var(--rose);
}
[data-testid="stSidebar"] [data-testid="stExpander"]:has([class*="st-key-mood"]) summary p { color: var(--rose); }

/* --- recency panel: butter accent (butter = "newness"; it matches the Era fit bar) ----- */
[data-testid="stSidebar"] [data-testid="stExpander"]:has([class*="st-key-recency"]) {
  border-color: rgba(255,212,121,.45); box-shadow: inset 4px 0 0 var(--butter);
}
[data-testid="stSidebar"] [data-testid="stExpander"]:has([class*="st-key-recency"]) summary p { color: var(--butter); }
[data-testid="stSidebar"] h3 { margin-top: 0.5rem; }

/* --- sidebar: bigger labels, roomier groups -------------------------------- */
[data-testid="stSidebar"] h3 { font-size: 1.6rem; }
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
[data-testid="stSidebar"] label p { font-size: 1rem; font-weight: 600; }
[data-testid="stSidebar"] [data-testid="stExpander"] { border-radius: 16px; background: rgba(255,255,255,.03); border-color: var(--line); }
[data-testid="stSidebar"] [data-testid="stExpander"] summary { padding: 0.75rem 1rem; }
[data-testid="stSidebar"] [data-testid="stExpander"] summary p { font-size: 1.05rem; font-weight: 700; }
[data-testid="stSidebar"] [data-testid="stExpanderDetails"] { gap: 1.25rem; padding-top: 0.5rem; }
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] { padding-top: 1rem; }

/* --- result cards -------------------------------------------------------- */
[class*="st-key-card-"] {
  background: linear-gradient(160deg, var(--surface-2), var(--surface));
  border: 1px solid var(--line); border-radius: 24px; padding: 1.5rem;
  box-shadow: 0 10px 32px rgba(0,0,0,.28);
}
/* the top-ranked card (and only that card) gets the gradient border and glow */
[class*="st-key-card-top-"] {
  border: 2px solid transparent;
  background: linear-gradient(160deg, var(--surface-2), var(--surface)) padding-box, var(--grad) border-box;
  box-shadow: 0 14px 44px rgba(255,122,89,.22);
}
.tb-badge { display: inline-block; background: var(--grad); color: #2A1224; font-weight: 700; font-size: 0.875rem;
            padding: 0.25rem 0.75rem; border-radius: 999px; margin-bottom: 0.5rem; }
.tb-head { display: flex; justify-content: space-between; align-items: flex-start; gap: 1rem; }
.tb-title { font-family: "Fraunces", Georgia, serif; font-size: 1.7rem; font-weight: 600; line-height: 1.2; }
.tb-meta { color: var(--muted); font-size: 1.05rem; margin-top: 0.25rem; }
.tb-score { text-align: center; min-width: 4.5rem; flex-shrink: 0; white-space: nowrap; padding: 0.5rem 0.75rem; border-radius: 16px;
            background: rgba(255,138,101,.16); color: var(--coral); }
.tb-score b { display: block; font-size: 1.7rem; line-height: 1.1; }
.tb-score small { font-size: 0.875rem; font-weight: 600; letter-spacing: .04em; }
.tb-chips { margin: 0.75rem 0 0 0; }
.tb-chip { display: inline-block; padding: 0.25rem 0.875rem; margin: 0 0.5rem 0.5rem 0; border-radius: 999px;
           background: rgba(247,235,221,.10); font-size: 0.9rem; }
.tb-chip.stream { background: rgba(255,212,121,.16); color: var(--butter); }
.tb-chip.none { background: transparent; border: 1px dashed rgba(247,235,221,.3); color: var(--muted); }
.tb-chips .tb-meta { font-size: 0.9rem; margin-right: 0.25rem; }
.tb-plot { color: #E6D6E0; font-size: 1.05rem; margin: 0.5rem 0 0.5rem 0; line-height: 1.6; }

/* --- "why this pick" panel (sits to the right of the details) ---------------- */
.tb-why { background: rgba(20,12,22,.45); border: 1px solid var(--line); border-radius: 18px; padding: 1rem 1.25rem 1.25rem; }
.tb-why-title { font-size: 0.875rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; color: var(--muted); }
.tb-why-row { margin-top: 1rem; }
.tb-why-top { display: flex; justify-content: space-between; align-items: baseline; gap: 0.5rem; margin-bottom: 0.5rem; }
.tb-why-label { font-size: 1rem; font-weight: 700; display: flex; align-items: center; gap: 0.5rem; }
.tb-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
.tb-pts { font-size: 1rem; font-weight: 600; color: var(--text); font-variant-numeric: tabular-nums; }
.tb-bar { height: 8px; border-radius: 4px; background: rgba(247,235,221,.12); overflow: hidden; }
.tb-bar div { height: 100%; border-radius: 4px; }
.tb-why-text { color: var(--muted); font-size: 0.9rem; margin-top: 0.5rem; line-height: 1.45; }

/* --- posters ------------------------------------------------------------- */
div[data-testid="stImage"] img { border-radius: 16px; max-width: 280px !important; box-shadow: 0 8px 24px rgba(0,0,0,.4); }
.tb-noposter { aspect-ratio: 2 / 3; max-width: 280px; border-radius: 16px; border: 1px dashed rgba(247,235,221,.3);
               display: flex; align-items: center; justify-content: center; color: var(--muted); font-size: 0.95rem; }

/* --- footer (TMDB attribution) --------------------------------------------------------- */
.tb-footer { margin-top: 2rem; padding-top: 1.5rem; border-top: 1px solid var(--line); text-align: center;
             color: var(--muted); font-size: 0.9rem; line-height: 1.6; }
.tb-footer-title { font-size: 0.875rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; margin-bottom: 0.5rem; }
.tb-tmdb-logo { display: block; margin: 0 auto 0.75rem; height: 16px; width: auto; }  /* far smaller than the Tiebreaker wordmark: TMDB requires that */

/* --- narrower screens: the "why" panel drops below the details, then
       everything stacks on phones ----------------------------------------- */
@media (max-width: 1180px) {
  [class*="st-key-card-"] [data-testid="stHorizontalBlock"] { flex-wrap: wrap; }
  [class*="st-key-card-"] [data-testid="stColumn"]:nth-child(1) { flex: 0 0 190px !important; width: 190px !important; min-width: 0 !important; }
  [class*="st-key-card-"] [data-testid="stColumn"]:nth-child(2) { flex: 1 1 0 !important; width: auto !important; min-width: 0 !important; }
  [class*="st-key-card-"] [data-testid="stColumn"]:nth-child(3) { flex: 1 1 100% !important; width: 100% !important; min-width: 100% !important; }
}
@media (max-width: 640px) {
  .block-container { padding: 1.25rem 1rem 3rem; }
  .st-key-hero { padding: 1.5rem 1.25rem; border-radius: 22px; }
  .tb-wordmark { font-size: 2.6rem; }
  .st-key-hero::before { width: 380px; height: 285px; right: -110px; opacity: .10; }
  .tb-tagline { font-size: 1.05rem; }
  [class*="st-key-card-"] { padding: 1rem; border-radius: 20px; }
  [class*="st-key-card-"] [data-testid="stColumn"]:nth-child(1),
  [class*="st-key-card-"] [data-testid="stColumn"]:nth-child(2) { flex: 1 1 100% !important; width: 100% !important; min-width: 100% !important; }
  .tb-title { font-size: 1.5rem; }
  div[data-testid="stImage"] img, .tb-noposter { max-width: 170px !important; }
}
</style>
"""

# TMDB's API terms require (1) this notice, placed prominently, and (2) one of
# their approved logos, shown LESS prominently than our own branding. Posters
# and all film details in this app come from TMDB. The wording is theirs:
# "This [website, program, service, application, product] uses TMDB and the TMDB
# APIs but is not endorsed, certified, or otherwise approved by TMDB."
TMDB_NOTICE = ("This application uses TMDB and the TMDB APIs but is not endorsed, "
               "certified, or otherwise approved by TMDB.")
# The approved "short" logo downloaded from themoviedb.org/about/logos-attribution.
TMDB_LOGO_FILE = Path(__file__).parent / "assets" / "tmdb-logo.svg"

# Two overlapping rings: two people, one shared choice.
LOGO = (
    '<svg width="56" height="56" viewBox="0 0 44 44" fill="none" aria-hidden="true">'
    '<circle cx="17" cy="22" r="13" stroke="#FF8A65" stroke-width="4"/>'
    '<circle cx="27" cy="22" r="13" stroke="#F27AA5" stroke-width="4"/></svg>'
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
def run_recommend(_movies, _features, size, query, tone, recency, candidate_ids, n):
    """(top n results, how many films matched in total)"""
    return recommender.recommend_with_count(
        query or None, tone=tone, recency=recency, n=n, candidate_ids=candidate_ids,
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
        "mood_on": False, "mood": 0.0, "recency": 0.0,
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
    """Draw the sidebar. Returns (criteria for filters.py, wanted tone or None, recency)."""
    ss = st.session_state
    st.sidebar.button("Reset all", on_click=reset_filters, args=(defaults,), width="stretch")

    # Two clearly separate groups. Preferences only RE-ORDER the results (no film
    # is ever removed by them); filters REMOVE films that don't fit.
    st.sidebar.markdown("### Preferences")
    st.sidebar.caption("Nudges. They re-order your results but never remove a film.")

    with st.sidebar.expander("Mood", expanded=True):
        st.toggle("Match a mood", key="mood_on",
                  help="Prefer movies whose plot reads darker or lighter. Mood words in your search "
                       "(dark, light, gritty...) are then ignored so the mood isn't counted twice.")
        if ss["mood_on"]:
            st.select_slider("Dark to light", options=MOOD_STEPS, key="mood",
                             format_func=lambda v: f"{tone_label(v).capitalize()} ({v:+.1f})")

    with st.sidebar.expander("Newer or older?", expanded=True):
        st.select_slider("Recency", options=list(RECENCY_LABELS), key="recency",
                         format_func=RECENCY_LABELS.get,
                         help="A nudge, not a filter: films from the other era are still shown, just lower "
                              "down. To hide films by year, use Release date under Filters.")

    st.sidebar.markdown("### Filters")
    st.sidebar.caption("Hard cuts. Films that don't fit are removed.")

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
        st.slider("Release year", options["year_min"], options["year_max"], key="years",
                  help="Removes films outside this range. To lean newer or older without removing "
                       "anything, use Recency under Preferences.")
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
    return criteria, tone, ss["recency"]


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
def why_rows(result, query, tone, recency=0.0):
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

    if parts.get("recency") is not None:
        year = result["movie"].get("release_year")
        wanted = "newer films" if recency > 0 else "classics"
        text = (f"You asked for {wanted}; this came out in {year}." if year
                else f"You asked for {wanted}; its release year is unknown, so it counts as average.")
        rows.append(("recency", "Era fit", parts["recency"], text))

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
            f'<div class="tb-why-row"><div class="tb-why-top"><span class="tb-why-label">'
            f'<span class="tb-dot" style="background:{row["colour"]}"></span>{esc(row["label"])}</span>'
            f'<span class="tb-pts">{points}</span></div>'
            f'<div class="tb-bar"><div style="width:{max(row["score"], 0) * 100:.0f}%;background:{row["colour"]}"></div></div>'
            f'<div class="tb-why-text">{esc(row["text"])}</div></div>'
        )
    return f'<div class="tb-why"><div class="tb-why-title">Why this pick</div>{body}</div>'


def render_card(movie, score, why_html, key, score_label="match", badge=None):
    """One result: poster, details, and the "why this pick" panel side by side.

    `badge` is set only for the top-ranked card. It switches on the gradient
    border (the CSS looks for "card-top" in the container's key).
    """
    with st.container(key=f"card-top-{key}" if badge else f"card-{key}"):
        poster, info, why = st.columns([1.2, 2.2, 1.4], gap="medium")
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
            badge_html = f'<div class="tb-badge">★ {esc(badge)}</div>' if badge else ""

            st.markdown(
                f'{badge_html}<div class="tb-head"><div><div class="tb-title">{esc(movie["title"])}</div>'
                f'<div class="tb-meta">{esc(" · ".join(meta))}</div></div>'
                f'<div class="tb-score"><b>{round(score * 100)}</b><small>{esc(score_label)}</small></div></div>'
                f'<div class="tb-chips">{genres}</div>'
                f'<div class="tb-chips"><span class="tb-meta">Streaming: </span>{streams}</div>'
                f'<div class="tb-plot">{esc(shorten(movie.get("overview")))}</div>',
                unsafe_allow_html=True,
            )
            st.button("More like this", key=key, on_click=open_more_like, args=(movie["id"],))
        with why:
            st.markdown(why_html, unsafe_allow_html=True)


def open_more_like(movie_id):
    st.session_state["more_like"] = movie_id
    st.session_state["scroll_top"] = True  # the new list should start at its heading


def close_more_like():
    st.session_state["more_like"] = None


def use_example(text):
    """An example button was pressed: put its text in the search box."""
    st.session_state["query"] = text
    st.session_state["more_like"] = None


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------
def tmdb_logo_html():
    """The TMDB logo as a small linked image, or "" if the file is missing.

    The SVG is embedded in the page as base64 text, so the footer doesn't depend
    on TMDB's servers being reachable.
    """
    try:
        data = base64.b64encode(TMDB_LOGO_FILE.read_bytes()).decode("ascii")
    except OSError:
        return ""
    return ('<a href="https://www.themoviedb.org" target="_blank" rel="noopener noreferrer">'
            f'<img class="tb-tmdb-logo" alt="TMDB logo" src="data:image/svg+xml;base64,{data}"></a>')


def render_footer():
    """Credits: TMDB's required logo and notice, on every screen."""
    st.markdown(
        '<div class="tb-footer"><div class="tb-footer-title">Credits</div>'
        f'{tmdb_logo_html()}<div>{esc(TMDB_NOTICE)}</div>'
        '<div>Film data and posters come from The Movie Database.</div></div>',
        unsafe_allow_html=True,
    )


def main():
    st.markdown(STYLE, unsafe_allow_html=True)
    # render_page() has several early exits (no results, "More like this", no data).
    # Calling the footer here, after it returns, draws it on every screen.
    # (Don't use st.stop() inside render_page: Streamlit drops anything drawn after it.)
    render_page()
    render_footer()


def render_page():
    if not tmdb_client.CACHE_PATH.exists():
        st.error("No movie data yet. In a terminal, run `python build_cache.py` (about 4 minutes), "
                 "then reload this page.")
        return

    movies, features = load_catalogue()
    size = len(movies)
    options = filter_options(movies, size)
    defaults = default_values(options)
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    st.session_state.setdefault("more_like", None)
    by_id = {m["id"]: m for m in movies}

    criteria, tone, recency = render_sidebar(options, defaults)

    # --- hero: wordmark, pitch, search box and one-tap example searches -------
    with st.container(key="hero"):
        st.markdown(
            f'<div class="tb-brand">{LOGO}<div class="tb-wordmark">Tiebreaker</div></div>'
            '<div class="tb-tagline">Two people, one couch, one pick. Tell us the vibe and we will settle it.</div>',
            unsafe_allow_html=True,
        )
        query = st.text_input(
            "What are you in the mood for?", key="query", on_change=close_more_like,
            placeholder="What's the vibe tonight?",
            label_visibility="collapsed",
        ).strip()
        st.markdown('<div class="tb-try">Not sure? Try one of these:</div>', unsafe_allow_html=True)
        for column, text in zip(st.columns(len(EXAMPLES)), EXAMPLES):
            column.button(text, key=f"example_{text}", on_click=use_example, args=(text,))

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
                        key=f"more_{rank}_{result['movie']['id']}", score_label="similar",
                        badge="Closest match" if rank == 1 else None)
        return

    results, total = run_recommend(movies, features, size, query, tone, recency, candidate_ids, RESULTS)

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
        with st.container(key="results-head"):
            st.subheader("Top picks tonight")
            ranked_by = ["quality"] + (["mood"] if tone is not None else []) + (["release date"] if recency else [])
            st.caption("Type what you are in the mood for, or tap an idea above, to personalise the list. "
                       f"Right now it ranks by {' and '.join(ranked_by)}.")

    # With no search and no mood there is nothing to "match", only quality to rank by
    label = "match" if (query or tone is not None or recency) else "score"

    # Only films that match the search are ever shown (no filler), so say so when
    # there are fewer than a full page, or none.
    searched = bool(query) and recommender.query_has_matches(query, movies, features, tone)
    if searched and not results:
        st.info(f"No films match “{query}” with your current filters. Try loosening a filter, "
                "or searching for something a little broader.")
    elif searched and total < RESULTS:
        st.info(f"Only {total} film{'s' if total != 1 else ''} match “{query}” with your current filters, "
                "so that's all we're showing.")

    for rank, result in enumerate(results, 1):
        render_card(result["movie"], result["score"], render_why(why_rows(result, query, tone, recency)),
                    key=f"more_{rank}_{result['movie']['id']}", score_label=label,
                    badge="Top pick" if rank == 1 else None)


main()
