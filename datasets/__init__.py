"""
Dataset utilities for PPB benchmark runners.

Provides download and parsing helpers for conversational datasets
(e.g. ShareGPT) used by the ``llama-server`` runner to generate
realistic inference workloads.
"""

from .sharegpt import download_sharegpt, load_sharegpt_prompts

__all__ = ["download_sharegpt", "load_sharegpt_prompts"]
