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
        if m1 == "X":  # strike
            return True, ""
        r1 = roll_value(m1)
        if m2 == "/":
            if m1 in ("X", "/"):
                return False, "Spare '/' must follow a non-strike first roll."
            if r1 < 0 or r1 > 9:
                return False, "First roll must be 0–9, '-', or 'F' before a spare."
            return True, ""
        if m2 == "":
            return True, ""  # incomplete frame is allowed
        r2 = roll_value(m2)
        if r1 + r2 > 10:
            return False, "Two-roll total in a normal frame cannot exceed 10."
        return True, ""

    # --- 10th frame rules ---
    if m1 == "X":
        if m2 == "/":
            return False, "In 10th: '/' cannot directly follow a strike. Use digit or 'X'."
        if m2 not in ("", None) and m3 == "/":
            r2 = roll_value(m2)
            if r2 > 9:
                return False, "Invalid spare after a 10 in second roll."
        return True, ""

    r1 = roll_value(m1)
    if m2 == "/":
        if r1 < 0 or r1 > 9:
            return False, "First roll must be 0–9, '-', or 'F' before a spare."
        return True, ""

    if m2 == "":
        return True, ""  # incomplete is fine
    r2 = roll_value(m2)
    if r1 + r2 > 10:
        return False, "In 10th: first two rolls cannot exceed 10 unless second is '/'."
    if m3 not in ("", None):
        return False, "No 3rd roll in 10th frame without a strike or spare."
    return True, ""

def flatten_rolls(frames: List[Tuple[str, str, str]]) -> List[int]:
    """Flatten frames into a list of roll pin counts for bonus lookups."""
    flat: List[int] = []
    for i, (m1, m2, m3) in enumerate(frames, start=1):
        tenth = (i == 10)
        # First roll
        if m1:
            flat.append(roll_value(m1))
        # Second roll
        if not tenth:
            if m1 != "X" and m2:
                prev = flat[-1] if flat else None
                flat.append(roll_value(m2, prev_roll=prev))
        else:
            if m2:
                prev = flat[-1] if flat else None
                flat.append(roll_value(m2, prev_roll=prev))
            if m3:
                prev2 = flat[-1] if flat else None
                flat.append(roll_value(m3, prev_roll=prev2))
    return flat

def compute_scores(frames: List[Tuple[str, str, str]]) -> Tuple[List[Optional[int]], List[Optional[int]]]:
    """Compute per-frame and cumulative scores. None = incomplete/not yet scorable."""
    # Guard: drop any stray invalid sentinels before scoring
    frames = [(
        ("" if a == "!" else a),
        ("" if b == "!" else b),
        ("" if c == "!" else c)
    ) for (a, b, c) in frames]

    frame_scores: List[Optional[int]] = []
    flat = flatten_rolls(frames)

    # Helper to fetch the sum of next n rolls starting from start_idx (bonus lookup)
    def next_n_sum(start_idx: int, n: int) -> Optional[int]:
        if start_idx + n <= len(flat):
            return sum(flat[start_idx:start_idx+n])
        return None

    r_cursor = 0  # cursor across flattened rolls while scoring frames 1–9
    for i, (m1, m2, m3) in enumerate(frames, start=1):
        tenth = (i == 10)
        if not m1:  # frame not started
            frame_scores.append(None)
            continue

        if not tenth:
            if m1 == "X":
                # Strike: 10 + next two rolls
                bonus = next_n_sum(r_cursor + 1, 2)
                if bonus is None:
                    frame_scores.append(None)
                else:
                    frame_scores.append(10 + bonus)
                r_cursor += 1  # only one roll consumed in flat for strike
            else:
                # Need second roll to score the frame
                if not m2:
                    frame_scores.append(None)
                    r_cursor += 1  # first roll exists in flat
                    continue
                if m2 == "/":
                    # Spare: 10 + next one roll
                    bonus = next_n_sum(r_cursor + 2, 1)
                    if bonus is None:
                        frame_scores.append(None)
                    else:
                        frame_scores.append(10 + bonus)
                    r_cursor += 2
                else:
                    # Open frame
                    r1 = roll_value(m1)
                    r2 = roll_value(m2)
                    frame_scores.append(r1 + r2)
                    r_cursor += 2
        else:
            # 10th frame: sum what exists if complete per 10th rules
            if m1 == "X":
                if not m2 or not m3:
                    frame_scores.append(None)
                else:
                    v1 = 10
                    v2 = roll_value(m2)
                    if m3 == "/":
                        # spare relative to v2 (only valid if m2 not 'X')
                        v3 = 10 - v2 if m2 != "X" else 10
                    else:
                        v3 = roll_value(m3, prev_roll=v2)
                    frame_scores.append(v1 + v2 + v3)
            else:
                if not m2:
                    frame_scores.append(None)
                elif m2 == "/":
                    if not m3:
                        frame_scores.append(None)
                    else:
                        v1 = roll_value(m1)
                        v2 = 10 - v1
                        v3 = roll_value(m3)
                        frame_scores.append(v1 + v2 + v3)
                else:
                    v1 = roll_value(m1)
                    v2 = roll_value(m2)
                    frame_scores.append(v1 + v2)

    # Cumulative totals
    cumulative: List[Optional[int]] = []
    running = 0
    for fs in frame_scores:
        if fs is None:
            cumulative.append(None)
        else:
            running += fs
            cumulative.append(running)

    return frame_scores, cumulative

# -------------------------
# UI: Title & Instructions
# -------------------------

st.title("🎳 Bowling Scoring — Standard Rules")
st.caption("Enter rolls using bowling notation: **X** (strike), **/** (spare), **-** (miss), **F** (foul = 0), or digits **0–9**.")

# -------------------------
# Sidebar: setup players
# -------------------------
with st.sidebar:
    st.header("Setup")
    n_players = st.number_input("How many players?", min_value=1, max_value=10, value=2, step=1)

    uploaded = st.file_uploader("Optional: Upload names file (TXT/CSV, one name per line)", type=["txt", "csv"])
    uploaded_names: List[str] = []
    if uploaded:
        try:
            raw = uploaded.read()
            txt = raw.decode("utf-8", errors="ignore")
            if uploaded.name.lower().endswith(".csv"):
                # Simple CSV: take first column per line
                for line in io.StringIO(txt):
                    name = line.strip().split(",")[0].strip()
                    if name:
                        uploaded_names.append(name)
            else:
                for line in io.StringIO(txt):
                    name = line.strip()
                    if name:
                        uploaded_names.append(name)
        except Exception:
            st.warning("Could not read the uploaded file. Please check encoding/format.")

    st.markdown("**Player Names**")
    names: List[str] = []
    for i in range(int(n_players)):
        default_name = uploaded_names[i] if i < len(uploaded_names) else f"Player {i+1}"
        names.append(st.text_input(f"Name {i+1}", value=default_name, key=f"name_{i}"))

st.divider()

# ----------------------------------
# Main: per-player input & scoring UI
# ----------------------------------

FRAME_LABELS = [f"F{i}" for i in range(1, 11)]

def render_player_inputs(pid: int, pname: str):
    st.subheader(f"🧑‍💼 {pname}")
    st.caption("For frames 1–9, use two inputs (leave 2nd blank after a strike). For the 10th, up to three inputs apply.")

    # Header row
    cols = st.columns(11, gap="small")
    cols[0].markdown("**Frame**")
    for i, label in enumerate(FRAME_LABELS, start=1):
        cols[i].markdown(f"**{label}**")

    # Row 1: Roll 1
    r1_cols = st.columns(11, gap="small")
    r1_cols[0].markdown("_Roll 1_")
    # Row 2: Roll 2
    r2_cols = st.columns(11, gap="small")
    r2_cols[0].markdown("_Roll 2_")
    # Row 3: Roll 3 (10th frame only)
    r3_cols = st.columns(11, gap="small")
    r3_cols[0].markdown("_Roll 3 (10th)_")

    marks: List[Tuple[str, str, str]] = []
    has_invalid = False

    for f in range(1, 11):
        k1 = f"p{pid}_f{f}_r1"
        k2 = f"p{pid}_f{f}_r2"
        k3 = f"p{pid}_f{f}_r3"

        r1_raw = r1_cols[f].text_input("", key=k1, placeholder="X / - 0–9 F")
        r2_raw = r2_cols[f].text_input("", key=k2, placeholder=" / - 0–9 F")
        r3_raw = r3_cols[f].text_input("", key=k3, placeholder="X / - 0–9 F") if f == 10 else ""

        r1 = normalize_mark(r1_raw)
        r2 = normalize_mark(r2_raw)
        r3 = normalize_mark(r3_raw) if f == 10 else ""

        # Validate per-frame; display errors only if something was typed
        ok, msg = validate_frame_marks(
            r1 if r1 != "!" else r1_raw,
            r2 if r2 != "!" else r2_raw,
            r3 if (f == 10 and r3 != "!") else r3_raw,
            is_tenth=(f == 10)
        )
        if not ok and any([r1_raw.strip(), r2_raw.strip(), (r3_raw.strip() if f == 10 else "")]):
            st.error(f"Frame {f}: {msg}")
            has_invalid = True

        marks.append((r1, r2, r3))

    # If any invalid entries, don't attempt to score
    if has_invalid:
        st.warning("Please fix the highlighted frame inputs to compute scores.")
        df = pd.DataFrame({
            "Frame": list(range(1, 11)),
            "R1": [("" if m[0] == "!" else m[0]) for m in marks],
            "R2": [("" if m[1] == "!" else m[1]) for m in marks],
            "R3": [("" if m[2] == "!" else m[2]) for m in marks],
            "Frame Score": [None]*10,
            "Cumulative": [None]*10,
        })
        st.dataframe(df, hide_index=True, use_container_width=True)
        st.info(f"**{pname} — Total: (incomplete)**")
        return [None]*10, [None]*10, None

    # Sanitize any "!" sentinels before scoring (treat as empty)
    marks_clean: List[Tuple[str, str, str]] = []
    for (a, b, c) in marks:
        a = "" if a == "!" else a
        b = "" if b == "!" else b
        c = "" if c == "!" else c
        marks_clean.append((a, b, c))

    # Compute scores with safety net
    try:
        frame_scores, cumulative = compute_scores(marks_clean)
    except Exception as e:
        st.error("An internal scoring error occurred. Please check inputs.")
        st.exception(e)
        frame_scores, cumulative = [None]*10, [None]*10

    # Display per-frame & cumulative
    df = pd.DataFrame({
        "Frame": list(range(1, 11)),
        "R1": [m[0] for m in marks_clean],
        "R2": [m[1] for m in marks_clean],
        "R3": [m[2] for m in marks_clean],
        "Frame Score": frame_scores,
        "Cumulative": cumulative,
    })
    st.dataframe(df, hide_index=True, use_container_width=True)

    total = cumulative[-1] if cumulative and cumulative[-1] is not None else None
    if total is not None:
        st.success(f"**{pname} — Total: {total}**")
    else:
        st.info(f"**{pname} — Total: (incomplete)**")

    return frame_scores, cumulative, total

# Render all players and capture summary
summary_rows = []
for pid, pname in enumerate(names):
    # Border is optional; remove if your Streamlit version doesn't support it
    with st.container():
        _, _, tot = render_player_inputs(pid, pname)
        summary_rows.append({
            "Player": pname,
            **{f"F{i}": "" for i in range(1, 11)},  # reserved for future per-frame summary
            "Total": (tot if tot is not None else "")
        })

# Totals summary
st.divider()
st.subheader("📊 Score Summary (Totals)")
summary_df = pd.DataFrame(summary_rows)
st.dataframe(summary_df, hide_index=True, use_container_width=True)

# -------------------------
# Instructions & Notes
# -------------------------
with st.expander("Scoring Notes & Allowed Inputs"):
    st.markdown("""
- **Marks**  
  • `X` = strike (10 pins)  
  • `/` = spare (frame total becomes 10)  
  • `-` = miss (0 pins)  
  • `F` = foul (0 pins)  
  • `0–9` = pins knocked down  

- **Frames 1–9**  
  • Strike: enter `X` in **Roll 1** and leave **Roll 2** blank.  
  • Spare: enter first roll (e.g., `7`
