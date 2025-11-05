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

st.set_page_config(page_title="Cadence & Power Stability", layout="wide")
st.title("Cadence & Power Stability Visualizer")

st.markdown(
    "Upload a **FIT** (Export Original) or **CSV** with columns like "
    "`time, cadence, watts`. The app will resample to 1 Hz, find stable windows, and plot."
)

uploaded = st.file_uploader("Upload FIT or CSV", type=["fit", "csv"])

# Tunables (you can promote these to sidebar sliders)
CAD_WINDOW_S   = st.sidebar.number_input("Cadence window (s)", 10, 120, 30)
CAD_MIN_RPM    = st.sidebar.number_input("Cadence min RPM", 40, 120, 80)
CAD_MAX_CV     = st.sidebar.number_input("Cadence max CV", 0.01, 0.50, 0.08, step=0.01)
CAD_MAX_DROP   = st.sidebar.number_input("Cadence max dropout ratio", 0.0, 0.9, 0.10, step=0.05)
CAD_DROPOUT_RPM= st.sidebar.number_input("Cadence dropout ≤ RPM", 0, 30, 5)
CAD_MIN_SEG_S  = st.sidebar.number_input("Cadence min stable segment (s)", 5, 600, 100)

PWR_WINDOW_S   = st.sidebar.number_input("Power window (s)", 10, 120, 30)
PWR_MIN_W      = st.sidebar.number_input("Power min W", 0, 2000, 120)
PWR_MAX_CV     = st.sidebar.number_input("Power max CV", 0.01, 0.50, 0.12, step=0.01)
PWR_MAX_DROP   = st.sidebar.number_input("Power max dropout ratio", 0.0, 0.9, 0.10, step=0.05)
PWR_DROPOUT_W  = st.sidebar.number_input("Power dropout ≤ W", 0, 100, 5)
PWR_MIN_SEG_S  = st.sidebar.number_input("Power min stable segment (s)", 5, 600, 100)

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
        })
    df = (pd.DataFrame(rows).dropna(subset=["time"]).sort_values("time").reset_index(drop=True))
    return df

def read_csv_to_df(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(file_bytes))
    # try to parse time
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    else:
        st.error("CSV must contain a 'time' column.")
    return df

def detect_stable(series: pd.Series, index: pd.DatetimeIndex,
                  window_s: int, min_level: float, max_cv: float,
                  max_dropout: float, dropout_floor: float, min_seg_s: int):
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

if uploaded:
    suffix = Path(uploaded.name).suffix.lower()

    if suffix == ".fit":
        df = read_fit_to_df(uploaded.read())
    else:
        df = read_csv_to_df(uploaded.read())

    if not df.empty:
        # Ensure numeric columns exist
        for c in ["cadence", "watts"]:
            if c not in df.columns:
                df[c] = np.nan

        # 1 Hz normalization
        df1s = (df.set_index("time").resample("1S").mean())
        df1s["cadence"] = df1s["cadence"].clip(lower=0)
        df1s["watts"]   = df1s["watts"].clip(lower=0)
        df1s = df1s.reset_index()

        # Detect stable runs
        cad_runs = detect_stable(
            df1s["cadence"], df1s["time"],
            CAD_WINDOW_S, CAD_MIN_RPM, CAD_MAX_CV, CAD_MAX_DROP, CAD_DROPOUT_RPM, CAD_MIN_SEG_S
        )
        pwr_runs = detect_stable(
            df1s["watts"], df1s["time"],
            PWR_WINDOW_S, PWR_MIN_W, PWR_MAX_CV, PWR_MAX_DROP, PWR_DROPOUT_W, PWR_MIN_SEG_S
        )

        # Plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 6), sharex=True)

        # Cadence
        ax1.plot(df1s["time"], df1s["cadence"], linewidth=1, label="Cadence (rpm)")
        ax1.plot(df1s["time"], df1s["cadence"].rolling(15, min_periods=1, center=True).mean(),
                 linewidth=2, label="15s rolling avg")
        for stt, ent in cad_runs:
            ax1.axvspan(stt, ent, alpha=0.2, label="Stable cadence" if "Stable cadence" not in ax1.get_legend_handles_labels()[1] else None)
        ax1.set_ylabel("RPM")
        ax1.legend(loc="upper left")
        ax1.set_title("Cadence & Power with stable regions shaded")

        # Power
        ax2.plot(df1s["time"], df1s["watts"], linewidth=1, label="Power (W)")
        ax2.plot(df1s["time"], df1s["watts"].rolling(15, min_periods=1, center=True).mean(),
                 linewidth=2, label="15s rolling avg")
        for stt, ent in pwr_runs:
            ax2.axvspan(stt, ent, alpha=0.2, label="Stable power" if "Stable power" not in ax2.get_legend_handles_labels()[1] else None)
        ax2.set_ylabel("W"); ax2.set_xlabel("Time")
        ax2.legend(loc="upper left")

        st.pyplot(fig, clear_figure=True)

        # Summaries
        st.subheader("Stable Cadence Segments")
        if cad_runs:
            for i, (stt, ent) in enumerate(cad_runs, 1):
                seg = df1s.set_index("time")["cadence"].loc[stt:ent]
                st.write(f"{i:02d}) {stt} → {ent} | {int((ent-stt).total_seconds()+1)} s | mean {seg.mean():.1f} rpm | CV {(seg.std()/max(seg.mean(),1e-9)):.2%}")
        else:
            st.write("None")

        st.subheader("Stable Power Segments")
        if pwr_runs:
            for i, (stt, ent) in enumerate(pwr_runs, 1):
                seg = df1s.set_index("time")["watts"].loc[stt:ent]
                st.write(f"{i:02d}) {stt} → {ent} | {int((ent-stt).total_seconds()+1)} s | mean {seg.mean():.0f} W | CV {(seg.std()/max(seg.mean(),1e-9)):.2%}")
        else:
            st.write("None")
