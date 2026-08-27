"""Canonical multi-lingual normalization for clothing attributes, confirmation, and values.

Supports English, Urdu (script), and Roman Urdu inputs.
"""

from __future__ import annotations

import re
from typing import Any

COLOR_MAP: dict[str, str] = {
    "black": "black",
    "kaala": "black",
    "kala": "black",
    "سیاہ": "black",
    "کالا": "black",
    "blue": "blue",
    "neela": "blue",
    "nila": "blue",
    "نیلا": "blue",
    "white": "white",
    "safed": "white",
    "safaid": "white",
    "سفید": "white",
    "red": "red",
    "laal": "red",
    "lal": "red",
    "سرخ": "red",
    "لال": "red",
    "green": "green",
    "hara": "green",
    "sabz": "green",
    "سبز": "green",
    "ہرا": "green",
    "grey": "grey",
    "gray": "grey",
    "surmai": "grey",
    "سرمئی": "grey",
}

SIZE_MAP: dict[str, str] = {
    "small": "S",
    "s": "S",
    "chota": "S",
    "chhota": "S",
    "چھوٹا": "S",
    "medium": "M",
    "m": "M",
    "darmiyana": "M",
    "درمیانہ": "M",
    "large": "L",
    "l": "L",
    "bada": "L",
    "bara": "L",
    "بڑا": "L",
    "extra large": "XL",
    "xl": "XL",
    "bohot bada": "XL",
}

CONFIRMATION_MAP: dict[str, bool] = {
    "yes": True,
    "y": True,
    "haan": True,
    "han": True,
    "ji": True,
    "ji haan": True,
    "ہاں": True,
    "جی": True,
    "no": False,
    "n": False,
    "na": False,
    "nahi": False,
    "nahin": False,
    "نہیں": False,
    "نا": False,
}


def normalize_color(value: str) -> str:
    """Return canonical color name or lowercased string."""
    val = (value or "").strip().lower()
    return COLOR_MAP.get(val, val)


def normalize_size(value: str) -> str:
    """Return canonical size string (e.g. 'S', 'M', 'L', 'XL')."""
    val = (value or "").strip().lower()
    return SIZE_MAP.get(val, value.upper() if value else value)


def normalize_confirmation(value: str | bool) -> bool | None:
    """Normalize user expression into boolean confirmation status."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        val = value.strip().lower()
        return CONFIRMATION_MAP.get(val, None)
    return None
