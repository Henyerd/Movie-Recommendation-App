"""Run this once to download the movie data into data/movies.json.

Usage (from the project folder, with the virtual environment active):
    python build_cache.py            # build the cache if it doesn't exist
    python build_cache.py --force    # re-download even if it already exists
"""

import argparse
import os
from pathlib import Path

# Streamlit looks for .streamlit/secrets.toml relative to the current folder,
# so make sure we're in the project folder no matter where this was launched.
os.chdir(Path(__file__).parent)

import tmdb_client  # noqa: E402  (must come after the chdir above)

parser = argparse.ArgumentParser(description="Build the TMDB movie cache.")
parser.add_argument("--force", action="store_true", help="re-download even if cached")
args = parser.parse_args()

if tmdb_client.CACHE_PATH.exists() and not args.force:
    print(f"{tmdb_client.CACHE_PATH} already exists. Use --force to rebuild it.")
else:
    tmdb_client.build_cache()
