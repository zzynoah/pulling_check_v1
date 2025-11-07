# app.py
# Streamlit app: Cadence & Power Stability with Grade Gate + Overlap %
# ---------------------------------------------------------------
# Upload FIT or CSV -> resample 1 Hz -> detect stable cadence & power
# -> optional grade filter -> show overlap and percentage.

import io
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Optional FIT support
try:
    from fitparse import FitFile
    HAVE_FITPARSE = True
except Exception:
    HAVE_FITPARSE = False

st.set_page_config(page_title="Pulling (Cadence & Power Stability)", layout="wide")
st.title("Cadence + Power Stability → Overlap % (with grade gate)")

st.markdown(
    "Upload a **FIT** (Export Original) or **CSV** with columns like "
    "`time, cadence, watts` (and optionally `lat, lon, elev` or `grade`). "
    "The app resamples to 1 Hz, detects stable windows for cadence & power, "
    "supports excluding steep terrain, highlights the **overlap**, and reports the **percentage**."
)

# ---------------------- Sidebar Controls ----------------------
with st.sidebar:
    st.header("Cadence thresholds")
    CAD_WINDOW_S   = st.number_input("Window (s)", 10, 180, 30)
    CAD_MIN_RPM    = st.number_input("Min rpm", 40, 130, 80)
    CAD_MAX_CV     = st.number_input("Max CV (0–1)", 0.01, 0.50, 0.08, step=0.01, format="%.2f")
    CAD_MAX_DROP   = st.number_input("Max dropout ratio (0–1)", 0.0, 0.9, 0.10, step=0.05, format="%.2f")
    CAD_DROPOUT_RPM= st.number_input("Dropout ≤ rpm", 0, 50, 5)
    CAD_MIN_SEG_S  = st.number_input("Min segment (s)", 5, 900, 100)

    st.header("Power thresholds")
    PWR_WINDOW_S   = st.number_input("Window (s) ", 10, 180, 30)
    PWR_MIN_W      = st.number_input("Min watts", 0, 2000, 120)
    PWR_MAX_CV     = st.number_input("Max CV (0–1) ", 0.01, 0.50, 0.12, step=0.01, format="%.2f")
    PWR_MAX_DROP   = st.number_input("Max dropout ratio (0–1) ", 0.0, 0.9, 0.10, step=0.05, format="%.2f")
    PWR_DROPOUT_W  = st.number_input("Dropout ≤ W", 0, 200, 50)
    PWR_MIN_SEG_S  = st.number_input("Min segment (s) ", 5, 900, 100)

    st.header("Terrain (drafting gate)")
    use_grade_gate = st.checkbox("Exclude steep terrain (draft-poor)", value=True)
    uphill_max = st.number_input("Max uphill grade (%)", 0.0, 30.0, 5.0, step=0.5)
    downhill_max = st.number_input("Max downhill magnitude (%)", 0.0, 30.0, 7.0, step=0.5)
    gate_min_ratio = st.slider("Window gate coverage ≥", 0.5, 1.0, 0.8, 0.05)

    st.header("Percentage base")
    pct_base = st.radio("Compute % of…", ["Whole activity time", "Moving time (non-zero watts or cadence)"])

uploaded = st.file_uploader("Upload FIT or CSV", type=["fit", "csv"])

# ---------------------- Helpers ----------------------
def semicircles_to_deg(x):
    if pd.isna(x):
        return np.nan
    try:
        return float(x) * 180.0 / (2**31)
    except Exception:
        return np.nan

def read_fit_to_df(file_bytes: bytes) -> pd.DataFrame:
    if not HAVE_FITPARSE:
        st.error("fitparse is not installed on this deployment.")
        return pd.DataFrame()
    ff = FitFile(io.BytesIO(file_bytes))
    rows = []
    for rec in ff.get_messages("record"):
        v = rec.get_values()
        rows.append({
            "time":    v.get("timestamp"),
            "cadence": v.get("cadence"),
            "watts":   v.get("power"),
            "lat":     v.get("position_lat"),
            "lon":     v.get("position_long"),
            "elev":    v.get("altitude"),
            "grade":   v.get("grade")  # some devices provide it; often absent
        })
    df = pd.DataFrame(rows)
    if "time" in df:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    # Convert semicircles if necessary (heuristic: big absolute values)
    if "lat" in df and (df["lat"].dropna().abs().gt(1.5).any() or df["lon"].dropna().abs().gt(1.5).any()):
        df["lat"] = df["lat"].map(semicircles_to_deg)
        df["lon"] = df["lon"].map(semicircles_to_deg)
    return df

def read_csv_to_df(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(file_bytes))
    if "time" not in df.columns:
        st.error("CSV must contain a 'time' column.")
        return pd.DataFrame()
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    return df

def _haversine_m(lat1, lon1, lat2, lon2):
    # vectorized haversine in meters
    R = 6371000.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = phi2 - phi1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dphi/2.0)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dl/2.0)**2
    return 2.0 * R * np.arcsin(np.sqrt(a))

def ensure_grade(df1s: pd.DataFrame):
    """
    Ensure a 'grade' series exists (in %). If absent, derive from lat/lon/elev.
    Returns (df1s, grade_series_or_None).
    """
    if "grade" in df1s.columns:
        g = pd.to_numeric(df1s["grade"], errors="coerce")
        g = g.rolling(5, min_periods=1, center=True).median().clip(-30, 30)
        return df1s, g

    needed = {"lat", "lon", "elev"}
    if needed.issubset(set(df1s.columns)):
        lat = pd.to_numeric(df1s["lat"], errors="coerce")
        lon = pd.to_numeric(df1s["lon"], errors="coerce")
        ele = pd.to_numeric(df1s["elev"], errors="coerce")
        dist_m = pd.Series(0.0, index=df1s.index)
        dist_m.iloc[1:] = _haversine_m(lat.shift(), lon.shift(), lat, lon).iloc[1:]
        dele = ele.diff()
        eps = 0.5  # avoid div-by-zero; meters
        grade = 100.0 * (dele / np.maximum(dist_m, eps))
        grade = grade.rolling(5, min_periods=1, center=True).median().clip(-30, 30)
        return df1s, grade

    return df1s, None

def detect_stable(series: pd.Series, index: pd.DatetimeIndex,
                  window_s: int, min_level: float, max_cv: float,
                  max_dropout: float, dropout_floor: float, min_seg_s: int,
                  gate: pd.Series | None = None, gate_min_ratio: float = 0.8):
    """
    Detect 'stable' windows:
    - mean >= min_level
    - coefficient of variation <= max_cv
    - dropout ratio <= max_dropout (values <= dropout_floor treated as dropout)
    - optional gate: at least gate_min_ratio of each window must satisfy gate==1
    Returns list of (start_time, end_time).
    """
    s = pd.Series(series.values, index=index, dtype="float64").clip(lower=0)
    s_masked = s.where(s > dropout_floor)

    min_pts = int(window_s * 0.8)
    roll = s_masked.rolling(f"{window_s}s", min_periods=min_pts)
    mean_v = roll.mean()
    std_v  = roll.std()
    cv = (std_v / mean_v).replace([np.inf, -np.inf], np.nan)
    count_valid = roll.count()
    dropout_ratio = 1 - (count_valid / window_s)

    stable_mask = (mean_v >= min_level) & (cv <= max_cv) & (dropout_ratio <= max_dropout)

    if gate is not None:
        gate_i = gate.reindex(index).astype(float).fillna(0.0)
        gate_frac = gate_i.rolling(f"{window_s}s", min_periods=min_pts).mean()
        stable_mask = stable_mask & (gate_frac >= gate_min_ratio)

    runs = []
    if stable_mask.any():
        sm = stable_mask.astype(int)
        starts = (sm.diff().fillna(sm.iloc[0]) == 1)
        ends   = (sm.diff(-1).fillna(sm.iloc[-1]) == 1)
        st_times = stable_mask.index[starts]
        en_times = stable_mask.index[ends]
        for stt, ent in zip(st_times, en_times):
            dur = (ent - stt).total_seconds() + 1
            if dur >= min_seg_s:
                runs.append((stt, ent))
    return runs

def intersect_runs(runs_a, runs_b):
    i = j = 0
    out = []
    runs_a = sorted(runs_a); runs_b = sorted(runs_b)
    while i < len(runs_a) and j < len(runs_b):
        a0, a1 = runs_a[i]; b0, b1 = runs_b[j]
        st = max(a0, b0); en = min(a1, b1)
        if st <= en: out.append((st, en))
        if a1 < b1: i += 1
        else:       j += 1
    if not out: return out
    out.sort()
    merged = [out[0]]
    for st, en in out[1:]:
        lst, lend = merged[-1]
        if st <= lend + pd.Timedelta(seconds=1):
            merged[-1] = (lst, max(lend, en))
        else:
            merged.append((st, en))
    return merged

# ---------------------- Main ----------------------
if uploaded:
    suffix = Path(uploaded.name).suffix.lower()
    df = read_fit_to_df(uploaded.read()) if suffix == ".fit" else read_csv_to_df(uploaded.read())

    if not df.empty:
        # Ensure numeric columns exist
        for c in ["cadence", "watts", "lat", "lon", "elev", "grade"]:
            if c not in df.columns:
                df[c] = np.nan

        # 1 Hz normalization & cleaning
        df1s = (df.set_index("time").resample("1S").mean())
        df1s["cadence"] = df1s["cadence"].clip(lower=0)
        df1s["watts"]   = df1s["watts"].clip(lower=0)
        # Keep lat/lon/elev/grade if present
        for c in ["lat", "lon", "elev", "grade"]:
            if c in df1s.columns:
                df1s[c] = pd.to_numeric(df1s[c], errors="coerce")

        # Ensure/derive grade (in %)
        df1s, grade_series = ensure_grade(df1s)

        # Build grade gate: pass when grade within [-downhill_max, +uphill_max]
        grade_gate = None
        if use_grade_gate:
            if grade_series is None:
                st.warning("Grade filter enabled but grade is unavailable (need 'grade' or (lat, lon, elev)). Filter skipped.")
            else:
                grade_ok = (grade_series <= uphill_max) & (grade_series >= -downhill_max)
                grade_gate = grade_ok.astype(int)

        df1s = df1s.reset_index()

        # Stable windows (cadence & power)
        idx_time = df1s.set_index("time").index
        gate_for_roll = grade_gate.reindex(idx_time) if grade_gate is not None else None

        cad_runs = detect_stable(
            df1s["cadence"], df1s["time"],
            CAD_WINDOW_S, CAD_MIN_RPM, CAD_MAX_CV, CAD_MAX_DROP, CAD_DROPOUT_RPM, CAD_MIN_SEG_S,
            gate = gate_for_roll, gate_min_ratio = gate_min_ratio
        )
        pwr_runs = detect_stable(
            df1s["watts"], df1s["time"],
            PWR_WINDOW_S, PWR_MIN_W, PWR_MAX_CV, PWR_MAX_DROP, PWR_DROPOUT_W, PWR_MIN_SEG_S,
            gate = gate_for_roll, gate_min_ratio = gate_min_ratio
        )
        both_runs = intersect_runs(cad_runs, pwr_runs)

        # Percent base
        total_seconds = int((df1s["time"].iloc[-1] - df1s["time"].iloc[0]).total_seconds()) + 1
        if pct_base.startswith("Moving"):
            moving = ((df1s["watts"] > 0) | (df1s["cadence"] > 0)).astype(int)
            base_mask = moving.copy()
        else:
            base_mask = pd.Series(1, index=df1s.index)

        if use_grade_gate and (grade_gate is not None):
            gg = grade_gate.reindex(idx_time).reset_index(drop=True)
            base_mask = (base_mask.values & (gg.values == 1)).astype(int)

        base_seconds = int(base_mask.sum())
        both_seconds = sum(int((en - st).total_seconds()) + 1 for st, en in both_runs)
        pct_both = 100.0 * both_seconds / max(1, base_seconds)

        # ---- Plots ----
        df1s["cadence_smooth"] = df1s["cadence"].rolling(15, min_periods=1, center=True).mean()
        df1s["watts_smooth"]   = df1s["watts"].rolling(15, min_periods=1, center=True).mean()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 6), sharex=True)

        # Cadence
        ax1.plot(df1s["time"], df1s["cadence"], linewidth=1, label="Cadence (rpm)")
        ax1.plot(df1s["time"], df1s["cadence_smooth"], linewidth=2, label="15s rolling avg")
        for stt, ent in cad_runs:
            ax1.axvspan(stt, ent, alpha=0.2, label="Stable cadence" if "Stable cadence" not in ax1.get_legend_handles_labels()[1] else None)
        ax1.set_ylabel("RPM")
        ax1.set_title("Cadence & Power with stable regions shaded")
        ax1.legend(loc="upper left")

        # Power
        ax2.plot(df1s["time"], df1s["watts"], linewidth=1, label="Power (W)")
        ax2.plot(df1s["time"], df1s["watts_smooth"], linewidth=2, label="15s rolling avg")
        for stt, ent in pwr_runs:
            ax2.axvspan(stt, ent, alpha=0.2, label="Stable power" if "Stable power" not in ax2.get_legend_handles_labels()[1] else None)
        ax2.set_ylabel("W"); ax2.set_xlabel("Time")
        ax2.legend(loc="upper left")

        st.pyplot(fig, clear_figure=True)

        # Both-stable overlay
        fig2, ax = plt.subplots(1, 1, figsize=(13, 3))
        ax.plot(df1s["time"], df1s["cadence"], linewidth=1, alpha=0.6, label="Cadence (rpm)")
        ax.plot(df1s["time"], df1s["watts"],   linewidth=1, alpha=0.6, label="Power (W)")
        for stt, ent in both_runs:
            ax.axvspan(stt, ent, alpha=0.25, label="Both stable" if "Both stable" not in ax.get_legend_handles_labels()[1] else None)
        title_suffix = ""
        if use_grade_gate and (grade_gate is not None):
            title_suffix = f" (grade within [-{downhill_max:.1f}%, +{uphill_max:.1f}%], window ≥ {int(gate_min_ratio*100)}%)"
        ax.set_title("Overlap: BOTH cadence & power stable" + title_suffix)
        ax.set_xlabel("Time"); ax.set_ylabel("Value")
        ax.legend(loc="upper left")
        st.pyplot(fig2, clear_figure=True)

        # ---- Result: percentage ----
        st.subheader("Result")
        base_label = "moving" if pct_base.startswith("Moving") else "whole"
        st.metric(
            label=f"Both-stable time (% of {base_label} time)",
            value=f"{pct_both:.1f}%"
        )
        st.caption(
            f"Both-stable seconds: {both_seconds} / Base seconds: {base_seconds} "
            f"(Whole activity: {total_seconds} s)"
        )

        # Export intervals
        if both_runs:
            both_df = pd.DataFrame({"start":[st for st,_ in both_runs],
                                    "end":[en for _,en in both_runs]})
            st.download_button(
                "Download both-stable intervals (CSV)",
                both_df.to_csv(index=False).encode("utf-8"),
                file_name="both_stable_intervals.csv",
                mime="text/csv"
            )
