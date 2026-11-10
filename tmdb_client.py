"""TMDB API calls and caching.

This module is the only part of the app that talks to The Movie Database
(TMDB) website. It will request movie data (titles, genres, overviews,
reviews, etc.) and save the results into the data/ folder, so the app
doesn't have to re-download the same information every time it runs.
"""
