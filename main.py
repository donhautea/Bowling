# bowling_scoring_app.py
# Streamlit Bowling Scoring (Standard Rules)
# Run: streamlit run bowling_scoring_app.py

import streamlit as st
import pandas as pd
import io
import re

st.set_page_config(page_title="Bowling Scoring (Standard Rules)", layout="wide")

# -------------------------
# Helpers: parsing & scoring
# -------------------------

VALID_ROLL = re.compile(r"^(X|x|/|-|F|[0-9])?$")

def normalize_mark(mark: str) -> str:
    """Normalize user input to uppercase bowling mark or empty."""
    mark = (mark or "").strip()
    if not mark:
        return ""
    if not VALID_ROLL.match(mark):
        return "!"  # invalid sentinel
    if mark.lower() == "x":
        return "X"
    return mark.upper()

def roll_value(mark: str, prev_roll: int | None = None) -> int:
    """
    Convert a single mark to pin count.
    - 'X' = 10
    - '-' or 'F' = 0
    - '0'-'9' = int
    - '/' = spare: 10 - prev_roll (prev_roll must be 0..9)
    """
    if mark == "X":
        return 10
    if mark in ("-", "F", ""):
        return 0 if mark else 0
    if mark == "/":
        if prev_roll is None or prev_roll < 0 or prev_roll > 9:
            # Invalid spare context; handled by validator upstream
            return 0
        return 10 - prev_roll
    # digit
    return int(mark)

def validate_frame_marks(m1, m2, m3, is_tenth=False) -> tuple[bool, str]:
    """
    Validate marks for a frame under standard rules.
    Returns (is_valid, message).
    """
    # Basic character validity
    for m in (m1, m2, m3):
        if m and not VALID_ROLL.match(m):
            return False, "Use only X, /, -, F, or digits 0–9."

    if not is_tenth:
        # Frames 1–9
        if m1 == "X":  # strike; m2 and m3 should be blank (but we'll ignore if filled)
            return True, ""
        # If first is digit or '-'/'F', second can be '/' (spare) or digit/'-'
        r1 = roll_value(m1)
        if m2 == "/":
            if m1 in ("X", "/"):
                return False, "Spare '/' must follow a non-strike first roll."
            if r1 < 0 or r1 > 9:
                return False, "First roll must be 0–9, '-', or 'F' before a spare."
            return True, ""
        # Otherwise both numeric-ish
        if m1 == "X":
            return True, ""
        r2 = roll_value(m2)
        if r1 + r2 > 10:
            return False, "Two-roll total in a normal frame cannot exceed 10."
        return True, ""

    # 10th frame rules
    # Compute values sequentially with spare logic
    # Cases:
    #  - If first is 'X', up to two bonus rolls allowed.
    #  - Else if first + second is spare '/', one bonus roll allowed.
    #  - Else (open), third must be blank and first+second <= 10.
    if m1 == "X":
        # second can be 'X' or digits or '-'/'F' or '/'
        if m2 == "/":
            return False, "In 10th: '/' cannot directly follow a strike. Use digit or 'X'."
        # For m3, if m2 is digit/'-','F', m3 can be '/' (spare-on-bonus) or 'X' or digit
        if m2 not in ("X", "", None):
            r2 = roll_value(m2)
            if m3 == "/":
                if r2 > 9:
                    return False, "Invalid spare after a 10 in second roll."
        return True, ""

    # If first is not strike
    r1 = roll_value(m1)
    if m2 == "/":
        # allow one bonus (m3 any valid roll)
        if r1 < 0 or r1 > 9:
            return False, "First roll must be 0–9, '-', or 'F' before a spare."
        return True, ""
    # Otherwise open: two numeric-ish rolls, total <= 10, and m3 must be blank
    r2 = roll_value(m2)
    if r1 + r2 > 10:
        return False, "In 10th: first two rolls cannot exceed 10 unless second is '/'."
    if m3 not in ("", None):
        return False, "No 3rd roll in 10th frame without a strike or spare."
    return True, ""

def flatten_rolls(frames: list[tuple[str, str, str]]) -> list[int]:
    """Flatten frames into a list of roll pin counts for bonus lookups."""
    flat = []
    for i, (m1, m2, m3) in enumerate(frames, start=1):
        tenth = (i == 10)
        # First roll
        if m1:
            flat.append(roll_value(m1))
        # Second roll
        if not tenth:
            if m1 == "X":
                continue  # no second roll in frames 1–9 for strike
        if m2:
            prev = flat[-1] if flat else None
            flat.append(roll_value(m2, prev_roll=prev))
        # Third roll (10th frame only)
        if tenth and m3:
            if m2 == "/":
                # Spare occurred, third roll value is just its roll value (no spare math)
                flat.append(roll_value(m3))
            else:
                prev = flat[-1] if flat else None
                flat.append(roll_value(m3, prev_roll=prev))
    return flat

def compute_scores(frames: list[tuple[str, str, str]]) -> tuple[list[int | None], list[int | None]]:
    """
    Compute per-frame and cumulative scores.
    Returns (frame_scores, cumulative_scores), using None for incomplete/invalid scoring.
    """
    # Build a parallel index to map frame->indices in flat list
    rolls = []
    idx_map = []  # list of (start_index, num_rolls_used_for_frame_scoring_calc_view)
    for i, (m1, m2, m3) in enumerate(frames, start=1):
        tenth = (i == 10)
        start_idx = len(rolls)
        # push rolls with proper spare math at flatten time
        fr = []
        # First roll
        if m1:
            fr.append(roll_value(m1))
        # Second roll handling
        if not tenth:
            if m1 != "X" and m2:
                prev = fr[0] if fr else None
                fr.append(roll_value(m2, prev_roll=prev))
        else:
            if m2:
                prev = fr[0] if fr else None
                # Spare in 10th uses prev for math
                fr.append(roll_value(m2, prev_roll=prev))
            if m3:
                # For 10th, third roll may be spare relative to second (rare, but allowed only if second not X and not spare)
                prev2 = fr[-1] if fr else None
                fr.append(roll_value(m3, prev_roll=prev2))
        rolls.extend(fr)
        idx_map.append(start_idx)

    # Now calculate frame scores using bonus rules
    frame_scores: list[int | None] = []
    flat = flatten_rolls(frames)

    # A helper to fetch next n rolls for bonuses, or None if incomplete
    def next_n_sum(start_idx: int, n: int) -> int | None:
        if start_idx + n <= len(flat):
            return sum(flat[start_idx:start_idx+n])
        return None

    r_cursor = 0
    for i, (m1, m2, m3) in enumerate(frames, start=1):
        tenth = (i == 10)
        if not m1:  # not started
            frame_scores.append(None)
            continue

        if not tenth:
            if m1 == "X":
                # strike uses next two rolls as bonus
                bonus = next_n_sum(r_cursor + 1, 2)
                if bonus is None:
                    frame_scores.append(None)
                else:
                    frame_scores.append(10 + bonus)
                r_cursor += 1
            else:
                # two-roll frame
                if not m2:
                    frame_scores.append(None)
                    # do not advance cursor yet? We still advance by 0? We'll treat missing as 0 rolls counted so far.
                    # Instead, advance only for completed rolls actually in flat
                    r_inc = 0
                    if m1:
                        r_inc += 1
                    r_cursor += r_inc
                    continue
                if m2 == "/":
                    # spare: 10 + next one roll
                    bonus = next_n_sum(r_cursor + 2, 1)
                    if bonus is None:
                        frame_scores.append(None)
                    else:
                        frame_scores.append(10 + bonus)
                    r_cursor += 2
                else:
                    # open
                    r1 = roll_value(m1)
                    r2 = roll_value(m2)
                    frame_scores.append(r1 + r2)
                    r_cursor += 2
        else:
            # 10th frame: sum of its rolls
            # Need to know if it's complete:
            if m1 == "X":
                # Needs m2 and m3 present
                if not m2 or not m3:
                    frame_scores.append(None)
                else:
                    # Sum of the three actual rolls
                    # Careful: if m2 is '/', it's invalid case handled by validator
                    v1 = 10
                    v2 = roll_value(m2)
                    # if m2 not X and m3 is '/', spare on bonus relative to v2
                    if m3 == "/":
                        v3 = 10 - v2 if m2 not in ("X",) else 10  # safety
                    else:
                        v3 = roll_value(m3, prev_roll=v2)
                    frame_scores.append(v1 + v2 + v3)
            else:
                if not m2:
                    frame_scores.append(None)
                elif m2 == "/":
                    # needs third roll
                    if not m3:
                        frame_scores.append(None)
                    else:
                        v1 = roll_value(m1)
                        v2 = 10 - v1
                        v3 = roll_value(m3)
                        frame_scores.append(v1 + v2 + v3)
                else:
                    # open: must not have third
                    v1 = roll_value(m1)
                    v2 = roll_value(m2)
                    frame_scores.append(v1 + v2)

            # advance cursor by however many of the 10th rolls exist in flat up to now
            # but we won't use r_cursor after 10th anyway

    # cumulative
    cumu: list[int | None] = []
    running = 0
    for fs in frame_scores:
        if fs is None:
            cumu.append(None)
        else:
            running += fs
            cumu.append(running)
    return frame_scores, cumu

# -------------------------
# Sidebar: setup players
# -------------------------
st.title("🎳 Bowling Scoring — Standard Rules")
st.caption("Enter rolls using bowling notation: **X** (strike), **/** (spare), **-** (miss), **F** (foul = 0), or digits **0–9**.")

with st.sidebar:
    st.header("Setup")
    n_players = st.number_input("How many players?", min_value=1, max_value=10, value=2, step=1)

    uploaded = st.file_uploader("Optional: Upload names file (TXT/CSV, one name per line)", type=["txt", "csv"])
    uploaded_names: list[str] = []
    if uploaded:
        try:
            raw = uploaded.read()
            txt = raw.decode("utf-8", errors="ignore")
            # split by lines/commas
            if uploaded.name.lower().endswith(".csv"):
                # simple CSV: take first column
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
    names = []
    for i in range(int(n_players)):
        default_name = uploaded_names[i] if i < len(uploaded_names) else f"Player {i+1}"
        names.append(st.text_input(f"Name {i+1}", value=default_name, key=f"name_{i}"))

st.divider()

# ----------------------------------
# Main: per-player input & scoring UI
# ----------------------------------

# Build a structure to hold inputs per player & frame
FRAME_LABELS = [f"F{i}" for i in range(1, 11)]

def render_player_inputs(pid: int, pname: str):
    st.subheader(f"🧑‍💼 {pname}")
    st.caption("For frames 1–9, use two inputs (leave 2nd blank after a strike). For the 10th, up to three inputs apply.")
    # Table-like inputs
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

    marks = []
    for f in range(1, 11):
        k1 = f"p{pid}_f{f}_r1"
        k2 = f"p{pid}_f{f}_r2"
        k3 = f"p{pid}_f{f}_r3"

        r1 = normalize_mark(r1_cols[f].text_input("", key=k1, placeholder="X / - 0–9 F"))
        r2 = normalize_mark(r2_cols[f].text_input("", key=k2, placeholder=" / - 0–9 F"))
        r3 = ""
        if f == 10:
            r3 = normalize_mark(r3_cols[f].text_input("", key=k3, placeholder="X / - 0–9 F"))

        # Validate per-frame
        ok, msg = validate_frame_marks(r1, r2, r3, is_tenth=(f == 10))
        if not ok and any([r1, r2, r3]):
            st.error(f"Frame {f}: {msg}")
        marks.append((r1, r2, r3))

    # Compute scores
    frame_scores, cumulative = compute_scores(marks)

    # Display per-frame & cumulative
    df = pd.DataFrame({
        "Frame": list(range(1, 11)),
        "R1": [m[0] for m in marks],
        "R2": [m[1] for m in marks],
        "R3": [m[2] for m in marks],
        "Frame Score": frame_scores,
        "Cumulative": cumulative,
    })
    st.dataframe(df, hide_index=True, use_container_width=True)

    total = cumulative[-1] if cumulative[-1] is not None else None
    if total is not None:
        st.success(f"**{pname} — Total: {total}**")
    else:
        st.info(f"**{pname} — Total: (incomplete)**")

    return frame_scores, cumulative, total

# Render all players and capture summary
summary_rows = []
all_totals_ready = True

for pid, pname in enumerate(names):
    with st.container(border=True):
        _, _, tot = render_player_inputs(pid, pname)
        summary_rows.append({"Player": pname, **{f"F{i}": "" for i in range(1, 11)}, "Total": tot if tot is not None else ""})
        if tot is None:
            all_totals_ready = False

# Optional combined summary (totals)
st.divider()
st.subheader("📊 Score Summary (Totals)")
summary_df = pd.DataFrame(summary_rows)
st.dataframe(summary_df, hide_index=True, use_container_width=True)

# -------------------------
# Instructions & Notes
# -------------------------
with st.expander("Scoring Notes & Allowed Inputs"):
    st.markdown("""
- **Marks**:  
  • `X` = strike (10 pins)  
  • `/` = spare (frame total becomes 10)  
  • `-` = miss (0 pins)  
  • `F` = foul (0 pins)  
  • `0–9` = pins knocked down  

- **Frames 1–9**:  
  • Strike: enter `X` in **Roll 1** and leave **Roll 2** blank.  
  • Spare: enter first roll (e.g., `7`), then `/` in **Roll 2**.  
  • Open: enter two numbers or `-`/digit combinations; their sum must be ≤ 10.

- **10th Frame**:  
  • If **Roll 1** is `X`, you get two bonus rolls (enter them in **Roll 2** and **Roll 3**).  
  • If **Roll 1 + Roll 2** is a spare (use `/`), you get one bonus roll (use **Roll 3**).  
  • If open (no strike/spare), **Roll 3** must be left blank.

- **Validation** is enforced to follow **standard ten-pin bowling rules**. Totals update when a frame has enough information to score bonuses.
""")
