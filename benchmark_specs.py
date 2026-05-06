#!/usr/bin/env python3
"""Deterministic benchmark constants for public evaluation."""

from __future__ import annotations


PUBLIC_EVAL_SEEDS = [101, 202, 303, 404, 505, 606, 707, 808]


def public_eval_seeds(num_episodes: int) -> list[int]:
    if num_episodes <= len(PUBLIC_EVAL_SEEDS):
        return PUBLIC_EVAL_SEEDS[:num_episodes]
    seeds = list(PUBLIC_EVAL_SEEDS)
    while len(seeds) < num_episodes:
        seeds.append(seeds[-1] + 101)
    return seeds

