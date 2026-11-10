"""Scoring and ranking.

This module decides which movies to recommend. It will combine signals
(such as how similar a movie is to ones the user likes, and how positive
its reviews are) into a single score per movie, then sort the movies from
best match to worst.
"""
