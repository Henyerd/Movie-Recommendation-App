# Tiebreaker

**A movie recommender for two people who can't decide what to watch.**

Type the vibe you're after ("a tense heist", "funny and light", "animated for the whole family"), narrow things down with filters, and Tiebreaker ranks 1,000 well-known films and explains *why* each one made the list. It is a web app built with Python and [Streamlit](https://streamlit.io), using film data from [TMDB](https://www.themoviedb.org).

> This product uses the TMDB API but is not endorsed or certified by TMDB.

## What you can do

- **Search by vibe.** Free text like "scary space movie" or "romantic comedy". Informal words are understood ("funny" means comedy, "heist" also finds robbery and caper films).
- **Preferences** (nudges that re-order results but never remove a film)
  - **Mood:** a dark ↔ light slider, matched against the tone of each film's plot summary.
  - **Newer or older?:** lean toward new releases or toward classics. Off by default.
- **Filters** (hard cuts that remove films that don't fit): streaming service in Canada, genres (must have all / at least one / exclude), release year or decade, rating and vote count, runtime, age rating, original language, and actor or director.
- **"More like this"** on any result: finds the films most similar in genres, themes, cast and director.
- **Every card explains itself.** A "Why this pick" panel shows how much each factor added to the score.
- **Honest about gaps.** If only six films match your search, you get six, with a note saying so. Nothing unrelated is added to fill the page.

## How it works

1. **Data** (`tmdb_client.py`). A one-off script downloads about 1,000 popular and top-rated films from TMDB (details, cast, keywords, age ratings, Canadian streaming services, poster path) and saves them to `data/movies.json`. After that the app reads the file and never needs the network, except to load poster images.
2. **Text features** (`nlp.py`).
   - *TF-IDF* turns each film's genres, keywords, cast and director into numbers so two films can be compared. Fields are weighted 40% genre, 35% keywords, 15% cast, 10% director. This powers "More like this".
   - *VADER* sentiment scores the wording of each plot summary from −1 (dark) to +1 (light). This powers the Mood slider.
   - *Typed searches* are scored by the share of your words a film matches (a genre match counts fully, a keyword match slightly less). Informal words map straight to genres ("funny" is comedy), and related words such as heist and robbery count at 80%. A keyword only matches if your word is the whole keyword or its last word, so "diamond mine" does not match "diamond".
3. **Filters** (`filters.py`). Decide which films are allowed through. Several filters combine with AND; the options inside one filter combine with OR.
4. **Ranking** (`recommender.py`). Each allowed film gets a score from 0 to 1:
   - **55% content**: how well it matches your search
   - **25% mood**: how close its tone is to the one you chose (left out if mood is off)
   - **20% quality**: a fairness-adjusted rating (a Bayesian average, so 9.0 from 80 votes doesn't beat 8.2 from 50,000)
   - **recency** (optional, up to 30%): only when you move the Newer-or-older slider

   Components you didn't use are left out and the remaining weights are rescaled to add up to 1.
5. **Interface** (`app.py`). The Streamlit page: sidebar, search box, result cards, explanations.

### Limits worth knowing

- The catalogue is about 1,000 films, so niche searches can match only a handful.
- Matching relies on TMDB's genre and keyword tags, not on reading plots. A film with few tags is easy to miss.
- The mood score comes from the plot summary's wording, which is a rough signal.
- Actor search covers only the top five billed cast. Streaming availability is for Canada and covers subscription services only.

## Run it locally

You need **Python 3.12** (what the app is tested on; newer versions should also work) and a free TMDB account.

```bash
# 1. Get the code and create an isolated environment
git clone https://github.com/Henyerd/Movie-Recommendation-App.git
cd Movie-Recommendation-App
python3 -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**2. Add your own TMDB token.** The app's data comes from TMDB, and each TMDB key belongs to one account, so you need your own. Create a free account at [themoviedb.org](https://www.themoviedb.org), then go to *Settings → API* and copy your **API Read Access Token** (the long one that starts with `eyJ`). Create the file `.streamlit/secrets.toml`:

```toml
TMDB_TOKEN = "paste-your-read-access-token-here"
```

This file is git-ignored on purpose. Never commit it, and never paste the token into the code.

```bash
# 3. Download the film data once (about 4 to 5 minutes)
python build_cache.py

# 4. Start the app
streamlit run app.py
```

Step 3 writes `data/movies.json`. If you ever want a fresh copy, run `python build_cache.py --force`.

## Project layout

| File | What it does |
|---|---|
| `app.py` | The Streamlit interface |
| `recommender.py` | Scoring, blending and ranking |
| `nlp.py` | TF-IDF, VADER sentiment, query matching and synonyms |
| `filters.py` | The sidebar filters and the "why did nothing match?" explanations |
| `tmdb_client.py` | Talks to TMDB, handles rate limits, writes the cache |
| `build_cache.py` | Command-line script that builds `data/movies.json` |
| `.streamlit/config.toml` | Theme: colours, fonts and sizes |
| `.streamlit/secrets.toml` | **Your** TMDB token (git-ignored, you create it) |
| `data/movies.json` | The downloaded film data (git-ignored, built by you) |

## Deploying to Streamlit Community Cloud

1. Push the repo to GitHub, then create the app at [share.streamlit.io](https://share.streamlit.io).
2. In **Advanced settings**, choose **Python 3.12** from the Python version dropdown. This is the only place Community Cloud reads the version from; a `runtime.txt` file is not used.
3. In **Advanced settings → Secrets**, paste `TMDB_TOKEN = "..."` (the same line as in your local `secrets.toml`). The token is only used to *download* film data, so it is only needed if the deployed app builds its own data; an app that ships with a ready-made `data/movies.json` never calls TMDB.
4. **Provide the film data.** `data/movies.json` is git-ignored, so a fresh deployment has no film data until you add it (see the data notes below).

## Data, attribution and TMDB's terms

- Film data and posters come from TMDB. The app footer carries TMDB's required notice.
- TMDB's API terms allow **non-commercial** use only, and say cached data must not be kept longer than **6 months**. Rebuild the cache before then.
- Each person running their own copy should use their own TMDB token.
