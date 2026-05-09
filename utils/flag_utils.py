"""Convert structured llama.cpp flag dicts to CLI argument lists.

Flag dicts are the TOML-native representation of ``llama_cpp_args`` entries.
This module handles:

* Known single-dash short flags (e.g. ``ncmoe`` → ``-ncmoe 20``)
* Unknown keys mapped to double-dash long flags with underscores→dashes
  (e.g. ``flash_attn`` → ``--flash-attn``)
* Boolean ``True`` → flag only (no value); ``False`` → omit entirely
* The reserved ``extra_flags`` key → split and appended verbatim
* The reserved ``_label`` key → extracted as a human-readable label

Public API
----------
``parse_flag_entry(entry)`` → ``(flags_dict, label, extra_flags_raw)``
    Split a raw TOML entry dict into its three logical parts.

``build_extra_cli_args(flags, extra_flags_raw)`` → ``list[str]``
    Convert flags dict + raw string into a list of CLI tokens.

``expand_llama_cpp_args(explicit, range_spec)`` → ``list[dict]``
    Expand a range spec into individual flag-set dicts and merge with
    an explicit list.
"""

from __future__ import annotations

import itertools
from typing import Any

# ---------------------------------------------------------------------------
# Short-flag registry
# ---------------------------------------------------------------------------
# llama.cpp uses a mix of long (--flag) and short (-flag) argument styles.
# The flags listed here use a single dash (``-ncmoe 20``, ``-cmoe``).
# Unknown keys fall back to double-dash with underscore→dash conversion
# (``flash_attn`` → ``--flash-attn``).
#
# Keep this set in sync with the llama.cpp argument parser.  New flags can
# be added here as they are introduced upstream, but unknown keys already
# work via the ``--key-name`` fallback, so this list only affects dash style.
# ---------------------------------------------------------------------------

SHORT_FLAG_KEYS: frozenset[str] = frozenset(
    {
        # MoE / expert routing
        "ncmoe",   # -ncmoe N  — number of experts for continuous batching
        "cmoe",    # -cmoe     — use continuous batching for MoE models
        # Context / generation
        "c",       # -c N      — context size
        "n",       # -n N      — number of tokens to predict
        "p",       # -p N      — number of prompt tokens
        "b",       # -b N      — batch size
        # GPU
        "ngl",     # -ngl N    — GPU layers (same as --n-gpu-layers)
        "ts",      # -ts       — tensor split shorthand
        "mg",      # -mg N     — main GPU
        # Attention
        "fa",      # -fa       — enable flash attention
        # KV cache
        "nkvo",    # -nkvo     — disable KV cache offloading
        "ctk",     # -ctk TYPE — KV cache type (key)
        "ctv",     # -ctv TYPE — KV cache type (value)
        # Rope / scaling
        "rf",      # -rf FLOAT — rope frequency scale
        "rs",      # -rs FLOAT — rope scale
        # Threads
        "t",       # -t N      — threads for generation
        "tb",      # -tb N     — threads for prompt batch
        # Grp attn
        "gan",     # -gan N    — group attention factor N
        "gaf",     # -gaf N    — group attention width
        # Miscellaneous
        "rtr",     # -rtr      — process prompt and return timing results
        "sm",      # -sm       — split mode shorthand
    }
)

# Reserved metadata keys that are NOT passed to the CLI.
_RESERVED_KEYS: frozenset[str] = frozenset({"_label", "extra_flags"})


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def parse_flag_entry(entry: dict[str, Any]) -> tuple[dict[str, Any], str | None, str | None]:
    """Split a raw TOML flag entry into its logical components.

    Parameters
    ----------
    entry:
        A dict from the ``llama_cpp_args`` list, e.g.::

            {"ncmoe": 20, "cmoe": True, "_label": "ncmoe_20", "extra_flags": "-rtr"}

    Returns
    -------
    flags : dict
        Flag key→value pairs for CLI conversion (reserved keys removed).
    label : str | None
        Value of ``_label``, or ``None``.
    extra_flags_raw : str | None
        Value of ``extra_flags``, or ``None``.
    """
    label: str | None = entry.get("_label")
    extra_flags_raw: str | None = entry.get("extra_flags")
    flags = {k: v for k, v in entry.items() if k not in _RESERVED_KEYS}
    return flags, label, extra_flags_raw


def build_extra_cli_args(
    flags: dict[str, Any],
    extra_flags_raw: str | None = None,
) -> list[str]:
    """Convert a flags dict and optional raw string to a list of CLI tokens.

    Rules
    -----
    * Key in ``SHORT_FLAG_KEYS`` → single-dash prefix (``-ncmoe``)
    * Other keys → double-dash, underscores replaced by dashes (``--flash-attn``)
    * Value ``True`` → flag token only (boolean switch)
    * Value ``False`` → flag omitted entirely
    * Other values → flag token + str(value)
    * ``extra_flags_raw`` → split by whitespace, appended last

    Parameters
    ----------
    flags:
        Dict of CLI flags (metadata keys already removed).
    extra_flags_raw:
        Raw string of additional flags, e.g. ``"-rtr --some-flag 42"``.
        Appended after structured flags without interpretation.

    Returns
    -------
    list[str]
        Ordered list of CLI tokens ready to be appended to a command list.

    Examples
    --------
    >>> build_extra_cli_args({"ncmoe": 20, "cmoe": True, "fa": False})
    ['-ncmoe', '20', '-cmoe']
    >>> build_extra_cli_args({}, extra_flags_raw="-rtr")
    ['-rtr']
    >>> build_extra_cli_args({"flash_attn": True})
    ['--flash-attn']
    """
    tokens: list[str] = []

    for key, value in flags.items():
        if isinstance(value, bool):
            if not value:
                continue  # False → omit
            # True → flag only
            prefix = "-" if key in SHORT_FLAG_KEYS else "--"
            cli_key = key if key in SHORT_FLAG_KEYS else key.replace("_", "-")
            tokens.append(f"{prefix}{cli_key}")
        else:
            prefix = "-" if key in SHORT_FLAG_KEYS else "--"
            cli_key = key if key in SHORT_FLAG_KEYS else key.replace("_", "-")
            tokens.append(f"{prefix}{cli_key}")
            tokens.append(str(value))

    if extra_flags_raw:
        tokens.extend(extra_flags_raw.split())

    return tokens


def expand_llama_cpp_args(
    explicit: list[dict[str, Any]],
    range_spec: dict[str, Any],
) -> list[dict[str, Any]]:
    """Expand a range spec into individual flag-set dicts and merge with explicit entries.

    The ``range_spec`` maps flag names to ``{from, to, step}`` dicts.
    Each flag generates a list of integer values; the Cartesian product of
    all flag lists is computed, producing one dict per combination.  These
    are appended after ``explicit`` entries.

    Parameters
    ----------
    explicit:
        Entries listed directly under ``llama_cpp_args`` in the TOML.
    range_spec:
        Dict from ``[sweep.llama_cpp_args_range]``, e.g.::

            {"ncmoe": {"from": 20, "to": 99, "step": 10}}

    Returns
    -------
    list[dict]
        Combined list: explicit entries first, then range-generated entries.

    Examples
    --------
    >>> expand_llama_cpp_args([], {"ncmoe": {"from": 20, "to": 40, "step": 20}})
    [{'ncmoe': 20}, {'ncmoe': 40}]
    >>> expand_llama_cpp_args([{}], {})
    [{}]
    """
    if not range_spec:
        return list(explicit)

    flag_names: list[str] = []
    flag_value_lists: list[list[int]] = []

    for flag_name, spec in range_spec.items():
        if not isinstance(spec, dict):
            raise ValueError(
                f"Range spec for {flag_name!r} must be a dict with 'from' and 'to' keys; "
                f"got: {spec!r}"
            )
        start = int(spec["from"])
        stop = int(spec["to"])
        step = int(spec.get("step", 1))
        values = list(range(start, stop + 1, step))
        if not values:
            raise ValueError(
                f"Range spec for {flag_name!r} produced no values: "
                f"from={start}, to={stop}, step={step}"
            )
        flag_names.append(flag_name)
        flag_value_lists.append(values)

    range_combos: list[dict[str, Any]] = [
        dict(zip(flag_names, combo))
        for combo in itertools.product(*flag_value_lists)
    ]

    return list(explicit) + range_combos
