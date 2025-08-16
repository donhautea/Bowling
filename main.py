# bowling_scoring_app.py
# Streamlit Bowling Scoring (Standard Rules)
# Run: streamlit run bowling_scoring_app.py

import streamlit as st
import pandas as pd
import io
import re
from typing import List, Tuple, Optional

st.set_page_config(page_title="Bowling Scoring (Standard Rules)", layout="wide")

# -------------------------
# Helpers: parsing & scoring
# -------------------------

VALID_ROLL = re.compile(r"^(X|x|/|-|F|[0-9])?$")

def normalize_mark(mark: str) -> str:
    """Normalize user input to uppercase bowling mark or empty.
    Returns '!' as an invalid sentinel so the UI can flag errors (never scored)."""
    mark = (mark or "").strip()
    if not mark:
        return ""
    if not VALID_ROLL.match(mark):
        return "!"  # invalid sentinel for UI only (never passed to scorer)
    if mark.lower() == "x":
        return "X"
    return mark.upper()

def roll_value(mark: str, prev_roll: Optional[int] = None) -> int:
    """Convert a single mark to pin count.
    - 'X' = 10
    - '-' or 'F' or '' = 0
    - '0'-'9' = int(value)
    - '/' = spare: 10 - prev_roll (prev_roll must be 0..9)
    """
    if mark == "X":
        return 10
    if mark in ("-", "F", ""):
        return 0
    if mark == "/":
        if prev_roll is None or prev_roll < 0 or prev_roll > 9:
            # Invalid spare context; caller should have validated already.
            return 0
        return 10 - prev_roll
    # digit
    return int(mark)

def validate_frame_marks(m1: str, m2: str, m3: str, *, is_tenth: bool = False) -> Tuple[bool, str]:
    """Validate marks for a frame under standard rules. Returns (is_valid, message)."""
    # Disallow our internal invalid sentinel if it sneaks in
    if any(m == "!" for m in (m1, m2, m3)):
        return False, "Use only X, /, -, F, or digits 0–9."

    # Basic character validity
    for m in (m1, m2, m3):
        if m and not VALID_ROLL.match(m):
            return False, "Use only X, /, -, F, or digits 0–9."

    if not is_tenth:
        # Frames 1–9
        if m1 == "X":  # strike (second roll should be blank, but we'll be lenient)
            return True, ""
        # Not a strike: handle spare or open
        r1 = roll_value(m1)
        if m2 == "/":
            if m1 in ("X", "/"):
                return False, "Spare '/' must follow a non-strike first roll."
            if r1 < 0 or r1 > 9:
                return False, "First roll must be 0–9, '-', or 'F' before a spare."
            return True, ""
        # Open: two rolls, total <= 10
        if m2 == "":
            return True, ""  # incomplete is allowed; scoring will be pending
        r2 = roll_value(m2)
        if r1 + r2 > 10:
            return False, "Two-roll total in a normal frame cannot exceed 10."
        return True, ""

    # 10th frame rules
    if m1 == "X":
        # After a strike, '/' cannot directly follow in the second input
        if m2 == "/":
            return False, "In 10th: '/' cannot directly follow a strike. Use digit or 'X'."
        # If second is digit/miss/foul, third can be '/', digit, '-', 'F', or 'X'
        if m2 not in ("", None):
            if m3 == "/":
                r2 = roll_value(m2)
                if r2 > 9:
                    return False, "Invalid spare after a 10 in second roll."
        return True, ""

    # If first is not strike
    r1 = roll_value(m1)
    if m2 == "/":
        # spare: one bonus roll allowed (any mark)
        if r1 < 0 or r1 > 9:
            return False, "First roll must be 0–9, '-', or 'F' before a spare."
        return True, ""
    # Open: two numeric-ish rolls, total <= 10, and third must be blank
    if m2 == "":
        return True, ""  # incomplete is allowed
    r2 = roll_value(m2)
    if r1 + r2 > 10:
        return False, "In 10th: first two rolls cannot exceed 10 unless second is '/'."
    if m3 not in ("", None):
        return False, "No 3rd roll
