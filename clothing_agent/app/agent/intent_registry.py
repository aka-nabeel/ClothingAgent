"""Compatibility facade for the authoritative Fitzy intent registry.

Import this module when a caller historically expects a single registry module.
The implementation lives in `intents.py`; this file intentionally contains no
duplicate definitions.
"""
from .intents import (
    ActionName,
    IntentDefinition,
    IntentName,
    INTENT_REGISTRY,
    RequirementName,
    get_intent_definition,
    registered_intent_values,
    registered_intents,
)

__all__ = [
    "ActionName",
    "IntentDefinition",
    "IntentName",
    "INTENT_REGISTRY",
    "RequirementName",
    "get_intent_definition",
    "registered_intent_values",
    "registered_intents",
]
