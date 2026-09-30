"""Normalize harmless SSH/Windows paste wrappers, never discard pending input."""

import re


def clean_choice(raw):
    # Bracketed-paste begin/end and terminal colour sequences are invisible on
    # screen but otherwise make int('4') parsing fail on the first paste.
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", raw)
    return text.translate({ord(c): None for c in "\ufeff\u200b\u200c\u200d\x00"}).strip()
