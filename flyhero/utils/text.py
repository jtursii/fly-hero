"""Lenient text decoding shared by every reader of user-authored library text
files (song.ini, .chart). Real files in this library arrive in a mix of
encodings from different eras/tools of Clone Hero customs: UTF-8, UTF-8 with
a BOM, and outright UTF-16 (Windows Notepad "Unicode" saves)."""

from __future__ import annotations

from pathlib import Path

_UTF16_BOMS = (b"\xff\xfe", b"\xfe\xff")
_UTF8_BOM = b"\xef\xbb\xbf"


def read_text_lenient(path: Path) -> str:
    """Decode a text file, sniffing a leading BOM first (UTF-16 or UTF-8),
    then falling back through utf-8 / cp1252 / latin-1. latin-1 never raises,
    so this always returns something."""
    raw = path.read_bytes()
    if raw.startswith(_UTF16_BOMS):
        return raw.decode("utf-16")
    if raw.startswith(_UTF8_BOM):
        return raw.decode("utf-8-sig")
    for enc in ("utf-8", "cp1252"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace")
