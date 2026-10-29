"""GridSense EDA - exploratory analysis of KW (load) for ALL devices, to decide how/where to build load prediction.

Data comes from EITHER the files uploaded inside the app (one CSV at a time, up to 500 MB each - see .streamlit/config.toml)
OR a `data` folder made by prepare_data.py (data/devices.csv + data/device_<id>.parquet) placed next to this file.

Tabs
  Fleet overview   : one row per device (uptime, energy, sessions, predictability), when each device ran (device x day), plant total load
  Device deep-dive : timeline, WHEN kW is there / not there (hour x day, weekday x hour), sessions & idle/no-data gaps, distribution, predictability (ACF, persistence skill)
  Compare devices  : daily profiles, hourly overlay and correlation of any set of devices

"KW is there"  = KW above the running threshold (default 1 kW)   |   "idle" = measured but at/below threshold   |   "no data" = minute missing in the file
"""
import json
import os
import shutil
import tempfile
import time
import warnings
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import acf, adfuller

warnings.filterwarnings("ignore")
st.set_page_config(page_title="GridSense · Load EDA", page_icon="⚡", layout="wide", initial_sidebar_state="collapsed")

HERE = Path(__file__).resolve().parent
REPO_DATA = HERE / "data"                                   # data folder uploaded to GitHub (made by prepare_data.py) - optional
UP_DIR = Path(tempfile.gettempdir()) / "gridsense_upload"   # data built from files uploaded inside the app (lives on the server's disk, not in memory)
KEEP = ["KW", "RKW", "YKW", "BKW", "KVA", "PF", "I1", "I2", "I3", "V1", "V2", "V3", "F"]
POWER = ["KW", "RKW", "YKW", "BKW"]
LIMITS = {"KW": (-5e3, 5e3), "RKW": (-5e3, 5e3), "YKW": (-5e3, 5e3), "BKW": (-5e3, 5e3), "KVA": (0, 5e3), "PF": (-1.0001, 1.0001), "I1": (-1e4, 1e4), "I2": (-1e4, 1e4), "I3": (-1e4, 1e4),
          "V1": (0, 1000), "V2": (0, 1000), "V3": (0, 1000), "F": (0, 100)}      # outside = meter glitch -> empty


def data_dir():
    """Uploaded data wins over the GitHub `data` folder once at least one file has been uploaded."""
    return UP_DIR if (UP_DIR / "devices.csv").exists() else REPO_DATA
PHASES = ["RKW", "YKW", "BKW"]

# colour roles (used consistently everywhere)
INK, MUTED, GRID = "#0F172A", "#64748B", "#EDF0F6"
LOAD, FORECAST, GOOD, WARN, BAD, CYAN, VIOLET, BLUE = "#F59E0B", "#6366F1", "#10B981", "#F59E0B", "#F43F5E", "#06B6D4", "#8B5CF6", "#3B82F6"
GREY = MUTED
PHASE_COL = {"RKW": "#F43F5E", "YKW": "#F59E0B", "BKW": "#3B82F6"}

# ------------------------------------------------------------------ design system (light, forced - no config file needed)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600&display=swap');
:root { color-scheme: light; }
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] { background:#F5F7FB !important; color:#0F172A !important; font-family:'Inter',sans-serif; }
[data-testid="stHeader"] { background:transparent !important; } #MainMenu, footer { visibility:hidden; }
.block-container, [data-testid="stMainBlockContainer"] { max-width:100% !important; padding:3.4rem 2.4rem 3rem 2.4rem !important; }
.stApp p, .stApp label, .stApp li, [data-testid="stWidgetLabel"] p { color:#0F172A !important; font-family:'Inter',sans-serif; }
[data-testid="stCaptionContainer"] p { color:#64748B !important; }

/* app bar */
.appbar { display:flex; align-items:center; gap:.9rem; background:#FFFFFF; border:1px solid #E6EAF2; border-radius:18px; padding:.85rem 1.2rem; margin-bottom:1rem; box-shadow:0 1px 2px rgba(15,23,42,.04), 0 8px 24px -12px rgba(15,23,42,.10); }
.appbar .logo { width:42px; height:42px; border-radius:12px; display:grid; place-items:center; font-size:1.3rem; background:linear-gradient(135deg,#FBBF24,#F97316 55%,#F43F5E); box-shadow:0 6px 16px -6px rgba(249,115,22,.7); }
.appbar .t { font-weight:800; font-size:1.2rem; letter-spacing:-.4px; color:#0F172A !important; } .appbar .s { font-size:.78rem; color:#64748B !important; margin-top:-1px; }
.appbar .sp { flex:1; }
.chip { display:inline-flex; align-items:center; gap:.35rem; padding:.28rem .7rem; border-radius:999px; font-size:.76rem; font-weight:500; border:1px solid #E6EAF2; background:#F8FAFD; color:#475569 !important; }
.pill { display:inline-flex; align-items:center; gap:.45rem; padding:.3rem .8rem; border-radius:999px; font-size:.78rem; font-weight:600; }
.pill.run { background:#E8FAF3; color:#047857 !important; } .pill.idle { background:#F1F4F9; color:#475569 !important; } .pill.warn { background:#FFF4DD; color:#B45309 !important; } .pill.bad { background:#FFE9ED; color:#BE123C !important; }
.dot { width:8px; height:8px; border-radius:50%; background:currentColor; } .run .dot { animation:pulse 1.6s infinite; }
@keyframes pulse { 0%{box-shadow:0 0 0 0 rgba(16,185,129,.55);} 70%{box-shadow:0 0 0 8px rgba(16,185,129,0);} 100%{box-shadow:0 0 0 0 rgba(16,185,129,0);} }

/* tabs as a segmented pill nav */
.stTabs [data-baseweb="tab-list"] { gap:.25rem; background:#E9EDF5; padding:.3rem; border-radius:14px; width:fit-content; border:0; }
.stTabs [data-baseweb="tab"] { height:2.4rem; padding:0 1.15rem; border-radius:10px; background:transparent !important; color:#64748B !important; font-weight:600; }
.stTabs [aria-selected="true"] { background:#FFFFFF !important; color:#0F172A !important; box-shadow:0 1px 3px rgba(15,23,42,.12); }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display:none !important; }
.stTabs [data-baseweb="tab-panel"] { padding-top:1.1rem; }

/* cards */
[data-testid="stVerticalBlockBorderWrapper"], [class*="st-key-panel"] { background:#FFFFFF !important; border:1px solid #E6EAF2 !important; border-radius:18px !important; box-shadow:0 1px 2px rgba(15,23,42,.04), 0 10px 28px -16px rgba(15,23,42,.12); }
.ph { display:flex; align-items:baseline; gap:.6rem; margin:.1rem 0 .2rem; } .ph b { font-size:1rem; font-weight:700; color:#0F172A !important; letter-spacing:-.2px; } .ph span { font-size:.8rem; color:#64748B !important; }
.kpi { position:relative; overflow:hidden; background:#FFFFFF; border:1px solid #E6EAF2; border-radius:18px; padding:1rem 1.1rem .9rem; height:100%; box-shadow:0 1px 2px rgba(15,23,42,.04), 0 10px 28px -16px rgba(15,23,42,.12); }
.kpi .top { display:flex; align-items:center; gap:.55rem; } .kpi .ic { width:30px; height:30px; border-radius:9px; display:grid; place-items:center; font-size:.95rem; background:#F3F5FB; }
.kpi .l { color:#64748B !important; font-size:.8rem; font-weight:500; }
.kpi .v { font-family:'JetBrains Mono',monospace; font-size:1.75rem; font-weight:600; letter-spacing:-1px; margin-top:.55rem; color:#0F172A !important; }
.kpi .s { font-size:.78rem; margin-top:.1rem; color:#64748B !important; }
.kpi.good .v { color:#059669 !important; } .kpi.warn .v { color:#D97706 !important; } .kpi.bad .v { color:#E11D48 !important; } .kpi.info .v { color:#4F46E5 !important; }
.kpi .spark { position:absolute; right:0; bottom:0; width:46%; height:38px; opacity:.9; }
.hero-f { border-radius:18px; padding:1.3rem 1.4rem; height:100%; background:radial-gradient(420px 200px at 100% 0%, rgba(99,102,241,.16), transparent 70%), linear-gradient(180deg,#FFFFFF,#F7F8FF); border:1px solid #E3E6FA; }
.hero-f .l { font-size:.8rem; font-weight:600; color:#4F46E5 !important; text-transform:uppercase; letter-spacing:.06em; }
.hero-f .big { font-family:'JetBrains Mono',monospace; font-size:2.9rem; font-weight:600; letter-spacing:-2px; color:#0F172A !important; line-height:1.1; margin-top:.35rem; } .hero-f .big small { font-size:1.1rem; color:#64748B !important; letter-spacing:0; }
.hero-f .r { color:#475569 !important; font-size:.86rem; margin-top:.35rem; }
.badge { display:inline-block; padding:.18rem .55rem; border-radius:8px; font-size:.78rem; font-weight:600; margin-top:.6rem; } .badge.up { background:#E8FAF3; color:#047857 !important; } .badge.down { background:#FFE9ED; color:#BE123C !important; } .badge.flat { background:#F1F4F9; color:#475569 !important; }
.mini { display:flex; gap:1.3rem; margin-top:1rem; padding-top:.9rem; border-top:1px dashed #DDE2F3; } .mini div b { display:block; font-family:'JetBrains Mono',monospace; font-size:1rem; color:#0F172A !important; } .mini div span { font-size:.74rem; color:#64748B !important; }
.live { border-radius:18px; padding:1.2rem 1.3rem; background:linear-gradient(180deg,#FFFFFF,#FFFBF2); border:1px solid #F5E6C8; height:100%; }
.live .big { font-family:'JetBrains Mono',monospace; font-size:3.2rem; font-weight:600; letter-spacing:-2px; color:#0F172A !important; line-height:1; margin:.7rem 0 .2rem; } .live .big small { font-size:1.1rem; color:#64748B !important; }
.bar { margin:.55rem 0; } .bar .row { display:flex; justify-content:space-between; font-size:.8rem; color:#475569 !important; font-weight:500; } .bar .row b { font-family:'JetBrains Mono',monospace; color:#0F172A !important; }
.bar .trk { height:8px; background:#EEF1F7; border-radius:99px; margin-top:.25rem; overflow:hidden; } .bar .fil { height:100%; border-radius:99px; }
.note { border-left:3px solid #6366F1; background:#F4F5FF; padding:.7rem 1rem; border-radius:0 12px 12px 0; color:#475569 !important; font-size:.88rem; margin:.5rem 0 .9rem; } .note b, .note i { color:#1E293B !important; }
.note.amber { border-left-color:#F59E0B; background:#FFF8EB; }
.tblwrap { overflow-x:auto; border:1px solid #E6EAF2; border-radius:14px; background:#FFFFFF; margin:.4rem 0 1rem; }
table.tbl { border-collapse:collapse; width:100%; font-size:.84rem; } table.tbl th { text-align:right; background:#F7F9FC; color:#64748B !important; font-weight:600; padding:.55rem .8rem; border-bottom:1px solid #E6EAF2; white-space:nowrap; }
table.tbl td { text-align:right; padding:.5rem .8rem; border-bottom:1px solid #F0F3F8; color:#0F172A !important; font-family:'JetBrains Mono',monospace; font-size:.8rem; } table.tbl th:first-child, table.tbl td:first-child { text-align:left; }

/* inputs (forced light) */
[data-baseweb="select"] > div, [data-baseweb="input"], [data-baseweb="base-input"], input, [data-testid="stNumberInputContainer"] { background:#FFFFFF !important; color:#0F172A !important; border-color:#DDE2EE !important; border-radius:10px !important; }
[data-baseweb="popover"] *, ul[role="listbox"], li[role="option"] { background:#FFFFFF !important; color:#0F172A !important; } li[role="option"]:hover { background:#F1F3FF !important; }
[data-testid="stExpander"] { background:#FFFFFF; border:1px solid #E6EAF2 !important; border-radius:14px; } [data-testid="stExpander"] summary, [data-testid="stExpander"] summary p { color:#0F172A !important; font-weight:600; }
[data-testid="stBaseButton-secondary"] { background:#FFFFFF !important; color:#0F172A !important; border:1px solid #DDE2EE !important; border-radius:10px !important; font-weight:600; }
[data-testid="stBaseButton-primary"] { background:#0F172A !important; color:#FFFFFF !important; border:0 !important; border-radius:10px !important; font-weight:600; } [data-testid="stBaseButton-primary"] p { color:#FFFFFF !important; }
</style>""", unsafe_allow_html=True)


# ------------------------------------------------------------------ UI helpers
def spark(values, color):
    v = np.asarray(values, float); v = v[~np.isnan(v)]
    if len(v) < 2: return ""
    y = 27 - (v - v.min()) / (v.max() - v.min() + 1e-9) * 22
    pts = " ".join(f"{i * 100 / (len(v) - 1):.1f},{yy:.1f}" for i, yy in enumerate(y))
    return f'<svg class="spark" viewBox="0 0 100 30" preserveAspectRatio="none"><polygon points="0,30 {pts} 100,30" fill="{color}22"/><polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.8" vector-effect="non-scaling-stroke"/></svg>'


def kpi(label, value, sub="", tone="", icon="", trend=None, trend_color=FORECAST):
    st.markdown(f'<div class="kpi {tone}"><div class="top"><div class="ic">{icon}</div><div class="l">{label}</div></div><div class="v">{value}</div><div class="s">{sub}</div>'
                f'{spark(trend, trend_color) if trend is not None else ""}</div>', unsafe_allow_html=True)


def note(text, tone=""):
    st.markdown(f'<div class="note {tone}">{text}</div>', unsafe_allow_html=True)


def table(df, nd=3):
    html = df.round(nd).to_html(index=False, classes="tbl", border=0, na_rep="–")
    st.markdown('<div class="tblwrap">' + "".join(x.strip() for x in html.splitlines()) + "</div>", unsafe_allow_html=True)


@contextmanager
def panel(title, sub="", key="p"):
    with st.container(border=True, key=f"panel_{key}"):
        st.markdown(f'<div class="ph"><b>{title}</b><span>{sub}</span></div>', unsafe_allow_html=True)
        yield


def style(fig, h=440, legend=True, legend_y=1.02, top=95):
    """Title sits at the top-left; the legend gets its own row just above the plot, so they never overlap."""
    fig.update_layout(height=h + (top - 50), template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", hovermode="x unified",
                      font=dict(family="Inter, sans-serif", size=12.5, color="#334155"), margin=dict(l=10, r=10, t=top, b=10),
                      title=dict(x=0, xanchor="left", y=0.985, yanchor="top", font=dict(size=15, color=INK)),
                      hoverlabel=dict(bgcolor="#FFFFFF", bordercolor="#E2E8F0", font=dict(family="Inter, sans-serif", size=12, color=INK)),
                      legend=dict(orientation="h", x=0, xanchor="left", y=legend_y, yanchor="bottom", bgcolor="rgba(0,0,0,0)"), showlegend=legend)
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor="#E2E8F0", showspikes=False); fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def show(fig, key=None):
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key=key)




def hline(fig, y, text="", color=MUTED):
    fig.add_hline(y=y, line_dash="dot", line_color=color, annotation_text=text, annotation_position="top left", annotation_font=dict(size=11, color=color))


# ------------------------------------------------------------------ data loading
RUN_THRESHOLD_HELP = "Minutes with KW above this value count as 'KW is there' (machine running). Everything measured at or below it is 'idle'. Minutes missing in the file are 'no data'."
HZ = [(1, "1 min"), (5, "5 min"), (15, "15 min"), (30, "30 min"), (60, "1 h"), (120, "2 h"), (360, "6 h"), (1440, "24 h"), (10080, "7 days")]
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
CLASS_ORDER = ["Continuous (≥ 90 %)", "Mostly running (50–90 %)", "Intermittent (10–50 %)", "Rare bursts (< 10 %)"]
CLASS_COL = dict(zip(CLASS_ORDER, [GOOD, BLUE, LOAD, BAD]))


@st.cache_data(show_spinner=False)
def load_index():
    p = data_dir() / "devices.csv"
    if not p.exists(): return None
    return pd.read_csv(p, parse_dates=["first", "last"]).sort_values("DeviceID").reset_index(drop=True)


@st.cache_data(show_spinner=False, max_entries=6)
def load_device(dev):
    """One device on a COMPLETE one-minute grid: minutes missing in the file become NaN rows (= 'no data')."""
    d = pd.read_parquet(data_dir() / f"device_{dev}.parquet")
    d["Timestamps"] = pd.to_datetime(d["Timestamps"]).dt.floor("min")
    d = d.drop_duplicates("Timestamps", keep="last").set_index("Timestamps").sort_index()
    return d.reindex(pd.date_range(d.index.min(), d.index.max(), freq="min"))


def rle(state):
    """Run-length encoding -> (start index, end index (exclusive), state value) of every stretch."""
    cut = np.flatnonzero(np.diff(state)) + 1
    s, e = np.r_[0, cut], np.r_[cut, len(state)]
    return s, e, state[s]


def skill(x, lag):
    """Persistence forecast 'KW(t) = KW(t - lag)' vs. always predicting the overall mean. Returns (R², MAE)."""
    if len(x) <= lag + 200: return np.nan, np.nan
    a, b = x[lag:], x[:-lag]; m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 200: return np.nan, np.nan
    a, b = a[m], b[m]; sst = ((a - np.nanmean(x)) ** 2).sum()
    return (1 - ((a - b) ** 2).sum() / sst if sst > 0 else np.nan), float(np.abs(a - b).mean())


def hour_ranges(hours):
    hours = sorted(int(h) for h in hours)
    if not hours: return "none"
    out, a, b = [], hours[0], hours[0]
    for h in hours[1:]:
        if h == b + 1: b = h
        else: out.append((a, b)); a = b = h
    out.append((a, b))
    return ", ".join(f"{x:02d}:00–{(y + 1) % 24:02d}:00" for x, y in out)


def fmt_h(minutes):
    return f"{minutes / 60:,.1f} h" if minutes >= 90 else f"{minutes:,.0f} min"


def uptime_class(u):
    return CLASS_ORDER[0] if u >= 90 else CLASS_ORDER[1] if u >= 50 else CLASS_ORDER[2] if u >= 10 else CLASS_ORDER[3]


def verdict(s):
    """Rule-of-thumb hint on how this device can be forecast (based on the numbers in this table)."""
    if s["run_min"] < 300: return "Skip - too few running minutes"
    r30, r24 = s["r2_30"], s["r2_24h"]
    if s["uptime"] >= 90:
        return "Steady load - ARIMA on rolling mean should fit" if r30 >= .8 else "Steady but volatile - use a longer rolling window"
    if pd.notna(r24) and r24 >= .3 and r24 > r30: return "Daily on/off routine - add daily seasonality"
    if s["sessions_per_day"] > 6: return "Frequent on/off - model running-only data"
    return "On/off with long idle spells - use running-only data (KW > threshold)"


# ------------------------------------------------------------------ upload one CSV at a time (same cleaning as prepare_data.py)
def ingest(file, status):
    """Read one big History_SecureElite CSV in chunks, clean it and merge it into one parquet per device on disk. Memory use stays small."""
    UP_DIR.mkdir(parents=True, exist_ok=True); file.seek(0); parts, rows, t0 = {}, 0, time.time()
    for chunk in pd.read_csv(file, usecols=["DeviceID", "Timestamps"] + KEEP, chunksize=250_000, dtype={c: "float32" for c in KEEP}, encoding="utf-8-sig", low_memory=False):
        ts = pd.to_datetime(chunk["Timestamps"], format="%d-%m-%Y %H:%M:%S", errors="coerce")
        if ts.isna().mean() > .01: ts = pd.to_datetime(chunk["Timestamps"], dayfirst=True, errors="coerce")
        chunk["Timestamps"] = ts; chunk = chunk.dropna(subset=["Timestamps", "DeviceID"]); chunk["DeviceID"] = chunk["DeviceID"].astype("int32")
        for dev, g in chunk.groupby("DeviceID", sort=False): parts.setdefault(int(dev), []).append(g.drop(columns="DeviceID"))
        rows += len(chunk); status.update(label=f"Reading… {rows:,} rows ({time.time() - t0:.0f} s)")
    status.update(label=f"Merging {len(parts)} devices…")
    for dev, frames in parts.items():
        path = UP_DIR / f"device_{dev}.parquet"; new = pd.concat(frames, ignore_index=True)
        d = pd.concat([pd.read_parquet(path), new], ignore_index=True) if path.exists() else new
        d = d.sort_values("Timestamps").drop_duplicates("Timestamps", keep="last").reset_index(drop=True)      # same minute twice (file borders / re-upload) -> kept once
        for c, (lo, hi) in LIMITS.items(): d.loc[(d[c] < lo) | (d[c] > hi), c] = np.nan                       # glitches -> empty
        for c in POWER: d[c] = d[c].clip(lower=0)                                                              # negative kW noise -> 0
        d[KEEP] = d[KEEP].round(3).astype("float32"); d.to_parquet(path, index=False, compression="zstd")
    return rows


def rebuild_index(min_max_kw=1.0, min_rows=1000):
    """devices.csv for the uploaded data: only devices that really carried load (same rule as prepare_data.py)."""
    summary, skipped = [], []
    for p in sorted(UP_DIR.glob("device_*.parquet")):
        dev = int(p.stem.split("_")[1]); d = pd.read_parquet(p, columns=["Timestamps", "KW"]); kw = d.KW.dropna()
        if len(kw) < min_rows or kw.max() < min_max_kw: skipped.append(dev); continue
        summary.append(dict(DeviceID=dev, rows=len(d), first=d.Timestamps.min(), last=d.Timestamps.max(), avg_kw=round(float(kw.mean()), 2), max_kw=round(float(kw.max()), 1), file=p.name))
    pd.DataFrame(summary, columns=["DeviceID", "rows", "first", "last", "avg_kw", "max_kw", "file"]).to_csv(UP_DIR / "devices.csv", index=False)
    (UP_DIR / "skipped.json").write_text(json.dumps(skipped)); return len(summary), len(skipped)


def upload_section(has_data):
    log = UP_DIR / "files.json"; done = json.loads(log.read_text()) if log.exists() else []
    with st.expander(f"📤 Upload data files  ·  {len(done)} uploaded" if done else "📤 Upload data files", expanded=not has_data):
        st.markdown('<div class="note">Upload the <b>History_SecureElite…csv</b> files <b>one at a time</b> (each up to 500 MB). After every file the app cleans it, merges it with what is already there (same cleaning as prepare_data.py) '
                    'and refreshes all charts - you can start exploring after the first file. Uploading the same file twice does no harm.</div>', unsafe_allow_html=True)
        n = st.session_state.setdefault("up_n", 0)
        f = st.file_uploader("CSV file", type="csv", key=f"uploader_{n}", help="One file at a time. The upload limit is set in .streamlit/config.toml (maxUploadSize = 500).")
        if f is not None:
            with st.status(f"Processing {f.name}…", expanded=True) as status:
                try:
                    rows = ingest(f, status); nd, ns = rebuild_index(); done.append(f"{f.name}  ({rows:,} rows)"); log.write_text(json.dumps(done))
                    status.update(label=f"{f.name}: {rows:,} rows merged · {nd} devices with load, {ns} without", state="complete")
                except Exception as e:
                    status.update(label=f"Could not read {f.name}: {e}", state="error"); st.stop()
            st.cache_data.clear(); st.session_state["up_n"] = n + 1; st.rerun()
        if done:
            st.caption("Uploaded so far: " + " · ".join(done))
            if st.button("🗑 Remove all uploaded data and start again"):
                shutil.rmtree(UP_DIR, ignore_errors=True); st.cache_data.clear(); st.session_state["up_n"] = n + 1; st.rerun()
        elif (REPO_DATA / "devices.csv").exists(): st.caption("Right now the app shows the `data` folder from GitHub. Uploading a file here replaces it with your uploaded data.")


@st.cache_data(show_spinner=False)
def analyse(dev, thr):
    """Everything the app shows for one device. Computed once per (device, threshold)."""
    d = load_device(dev); t = d.index; kw = d["KW"].to_numpy(np.float64); n = len(d)
    nan = np.isnan(kw); run = ~nan & (kw > thr); idle = ~nan & ~run; valid = ~nan
    state = np.where(nan, 2, np.where(run, 1, 0)).astype(np.int8)
    s, e, v = rle(state); kz = np.where(nan, 0.0, kw); cs = np.r_[0.0, np.cumsum(kz)]; peaks = np.fmax.reduceat(kw, s); one = pd.Timedelta(minutes=1)

    def stretches(code):
        m = v == code; ss, ee = s[m], e[m]
        return pd.DataFrame({"start": t[ss], "end": t[ee - 1] + one, "minutes": ee - ss, "avg_kw": (cs[ee] - cs[ss]) / (ee - ss), "peak_kw": peaks[m], "kwh": (cs[ee] - cs[ss]) / 60})
    sessions, idles, nodata = stretches(1), stretches(0), stretches(2)
    for f in (sessions, idles, nodata): f["start_hour"] = f.start.dt.hour; f["weekday"] = f.start.dt.dayofweek

    base = pd.DataFrame({"kw": kw, "krun": np.where(run, kw, 0.0), "run": run.astype(float), "valid": valid.astype(float), "one": 1.0, "hour": t.hour, "dow": t.dayofweek, "day": t.normalize()})
    hg = base.groupby("hour"); hour = pd.DataFrame({"run_pct": hg.run.sum() / hg.valid.sum().replace(0, np.nan) * 100, "mean_kw": hg.kw.mean()}).reindex(range(24))
    rq = base[run].groupby("hour").kw.quantile([.1, .5, .9]).unstack() if run.any() else pd.DataFrame(columns=[.1, .5, .9], dtype=float)
    rq.columns = ["p10", "p50", "p90"]; hour = hour.join(rq)
    dg = base.groupby(["dow", "hour"]); dow_hour = (dg.run.sum() / dg.valid.sum().replace(0, np.nan) * 100).unstack().reindex(index=range(7), columns=range(24))
    cg = base.groupby(["day", "hour"]).agg(kw=("kw", "mean"), run=("run", "sum"), valid=("valid", "sum"), n=("one", "sum"))
    cal = dict(kw=cg.kw.unstack(), run=(cg.run / cg.valid.replace(0, np.nan) * 100).unstack(), miss=((cg.n - cg.valid) / cg.n * 100).unstack())
    gd = base.groupby("day").agg(kwsum=("kw", "sum"), krun=("krun", "sum"), run=("run", "sum"), valid=("valid", "sum"), n=("one", "sum"), peak=("kw", "max"))
    daily = pd.DataFrame({"kwh": gd.kwsum / 60, "run_h": gd.run / 60, "avg_run": gd.krun / gd.run.replace(0, np.nan), "missing_h": (gd.n - gd.valid) / 60, "peak": gd.peak})
    daily["weekend"] = daily.index.dayofweek >= 5

    sk = pd.DataFrame([(lab, *skill(kw, lag)) for lag, lab in HZ], columns=["horizon", "r2", "mae"])
    kr = kw[run]; both = run[1:] & run[:-1]; step = np.abs(np.diff(kw))[both]
    wk = daily.index.dayofweek < 5; wkd = base[run & (t.dayofweek < 5)].shape[0]; wke = base[run & (t.dayofweek >= 5)].shape[0]
    n_valid = int(valid.sum()); run_min = int(run.sum()); days = max(n / 1440, 1e-9)
    stats = dict(DeviceID=int(dev), first=t[0], last=t[-1], days=round(days, 1), minutes=n, coverage=valid.sum() / n * 100, run_min=run_min, idle_min=int(idle.sum()), nodata_min=int(nan.sum()),
                 uptime=run_min / n_valid * 100 if n_valid else 0.0, kwh=float(kz.sum() / 60), avg_kw=float(np.nanmean(kw)) if n_valid else np.nan, avg_run=float(kr.mean()) if run_min else np.nan,
                 median_run=float(np.median(kr)) if run_min else np.nan, p95_run=float(np.percentile(kr, 95)) if run_min else np.nan, peak=float(np.nanmax(kw)) if n_valid else np.nan,
                 sessions=len(sessions), sessions_per_day=len(sessions) / days, median_session_min=float(sessions.minutes.median()) if len(sessions) else np.nan,
                 longest_session_min=int(sessions.minutes.max()) if len(sessions) else 0, longest_idle_min=int(idles.minutes.max()) if len(idles) else 0, longest_gap_min=int(nodata.minutes.max()) if len(nodata) else 0,
                 first_run=t[run][0] if run_min else pd.NaT, last_run=t[run][-1] if run_min else pd.NaT, wd_run_h=wkd / 60, we_run_h=wke / 60,
                 step_pct=float(np.median(step) / kr.mean() * 100) if len(step) and run_min and kr.mean() > 0 else np.nan,
                 r2_30=float(sk.r2[3]), r2_60=float(sk.r2[4]), r2_24h=float(sk.r2[7]), r2_7d=float(sk.r2[8]))
    stats["class"] = uptime_class(stats["uptime"]); stats["verdict"] = verdict(stats)
    return dict(stats=stats, sessions=sessions, idles=idles, nodata=nodata, hour=hour, dow_hour=dow_hour, cal=cal, daily=daily, skill=sk, hourly=d["KW"].resample("1h").mean())


@st.cache_data(show_spinner="Analysing every device (only the first time)…")
def fleet(thr, ver):
    idx = load_index(); rows, daily, hrun, hkw, hourly = [], {}, {}, {}, {}
    for dev in idx.DeviceID:
        A = analyse(int(dev), thr); rows.append(A["stats"]); k = f"Device {int(dev)}"
        daily[k] = A["daily"].run_h; hrun[k] = A["hour"].run_pct; hkw[k] = A["hour"].mean_kw; hourly[k] = A["hourly"]
    tb = pd.DataFrame(rows)
    return dict(table=tb, daily=pd.DataFrame(daily), hrun=pd.DataFrame(hrun).T, hkw=pd.DataFrame(hkw).T, hourly=pd.DataFrame(hourly))


# ------------------------------------------------------------------ stationarity / autocorrelation (device page only)
@st.cache_data(show_spinner="Computing autocorrelation and stationarity test…")
def stationarity(dev):
    s = load_device(dev)["KW"].resample("10min").mean().interpolate(limit=6, limit_direction="both")
    x = s.fillna(s.mean()).to_numpy(float); out = dict(acf=None, adf_p=np.nan, adf_stat=np.nan, n=len(x))
    if len(x) > 300 and np.isfinite(x).all() and x.std() > 1e-6:
        out["acf"] = acf(x, nlags=int(min(1008, len(x) // 3)), fft=True)
        try: r = adfuller(x, maxlag=24, autolag="AIC"); out.update(adf_stat=float(r[0]), adf_p=float(r[1]))
        except Exception: pass
    return out


# ------------------------------------------------------------------ page
idx = load_index()
if idx is None or not len(idx):
    st.markdown('<div class="appbar"><div class="logo">⚡</div><div><div class="t">GridSense · Load EDA</div><div class="s">When is KW there, when is it not - for every device</div></div></div>', unsafe_allow_html=True)
    upload_section(False)
    st.info("Upload your first History_SecureElite CSV above to start.")
    st.stop()

ver = int(os.path.getmtime(data_dir() / "devices.csv"))
src_chip = "uploaded files" if data_dir() == UP_DIR else "GitHub data folder"
st.markdown(f"""<div class="appbar"><div class="logo">⚡</div><div><div class="t">GridSense · Load EDA</div><div class="s">When is KW there, when is it not - for every device</div></div><div class="sp"></div>
<span class="chip">{src_chip}</span><span class="chip">{len(idx)} devices</span><span class="chip">{idx['first'].min():%d %b %Y} → {idx['last'].max():%d %b %Y}</span></div>""", unsafe_allow_html=True)
upload_section(True)

c0 = st.columns([1.1, 3])
thr = c0[0].number_input("Running threshold (kW)", 0.0, 50.0, 1.0, 0.5, help=RUN_THRESHOLD_HELP)
c0[1].markdown('<div class="note" style="margin-top:1.6rem"><b>KW is there</b> = above the threshold · <b>idle</b> = measured, at or below it · <b>no data</b> = minute missing from the file. '
               'Devices that never carried load (max KW below 1) are left out.</div>', unsafe_allow_html=True)
F = fleet(float(thr), ver); T = F["table"]
LBL = {int(r.DeviceID): f"Device {int(r.DeviceID)}  ·  {r.uptime:.0f}% on  ·  avg run {0 if pd.isna(r.avg_run) else r.avg_run:,.0f} kW" for r in T.itertuples()}

tab_fleet, tab_dev, tab_cmp = st.tabs(["Fleet overview", "Device deep-dive", "Compare devices"])

# ================================================================== FLEET
with tab_fleet:
    k = st.columns(5)
    with k[0]: kpi("Devices analysed", f"{len(T)}", f"{int((T.run_min >= 300).sum())} with real running time", "info", "🏭")
    with k[1]: kpi("Energy · all devices", f"{T.kwh.sum() / 1000:,.1f} MWh", f"over {T.days.max():.0f} days", "", "🔋")
    with k[2]: kpi("Average plant load", f"{T.avg_kw.sum():,.0f} kW", "sum of every device's average", "", "📈")
    with k[3]: kpi("Continuous devices", f"{int((T['class'] == CLASS_ORDER[0]).sum())}", "on ≥ 90 % of measured minutes", "good", "🟢")
    with k[4]: kpi("Mostly idle devices", f"{int((T.uptime < 10).sum())}", "on < 10 % of measured minutes", "warn", "💤")
    st.write("")

    with panel("Device table", "click a column header to sort · the last column is a rule-of-thumb hint on how to forecast the device", key="ftable"):
        show_cols = {"DeviceID": "Device", "class": "Type", "uptime": "On %", "coverage": "Data %", "avg_run": "Avg kW (on)", "median_run": "Median kW (on)", "p95_run": "P95 kW", "peak": "Peak kW", "kwh": "Energy kWh",
                     "sessions": "Sessions", "sessions_per_day": "Sessions/day", "median_session_min": "Median session min", "longest_session_min": "Longest on min", "longest_idle_min": "Longest idle min",
                     "longest_gap_min": "Longest no-data min", "r2_30": "R² 30 min", "r2_24h": "R² 24 h", "verdict": "Forecast hint"}
        view = T[list(show_cols)].rename(columns=show_cols).sort_values("Avg kW (on)", ascending=False, na_position="last")
        st.dataframe(view, hide_index=True, width="stretch", height=420, column_config={
            "On %": st.column_config.ProgressColumn("On %", min_value=0, max_value=100, format="%.1f"), "Data %": st.column_config.ProgressColumn("Data %", min_value=0, max_value=100, format="%.1f"),
            "Avg kW (on)": st.column_config.NumberColumn(format="%.1f"), "Median kW (on)": st.column_config.NumberColumn(format="%.1f"), "P95 kW": st.column_config.NumberColumn(format="%.1f"),
            "Peak kW": st.column_config.NumberColumn(format="%.1f"), "Energy kWh": st.column_config.NumberColumn(format="%.0f"), "Sessions/day": st.column_config.NumberColumn(format="%.1f"),
            "R² 30 min": st.column_config.NumberColumn(format="%.2f", help="How well 'same as 30 minutes ago' predicts KW. 1 = perfect, ≤ 0 = useless."),
            "R² 24 h": st.column_config.NumberColumn(format="%.2f", help="How well 'same as at this time yesterday' predicts KW (daily routine strength).")})
        st.download_button("⬇ Device table (CSV)", T.to_csv(index=False).encode(), "device_summary_eda.csv", "text/csv", on_click="ignore")

    a, b = st.columns(2)
    with a, panel("Average load while running", "kW, coloured by how much of the time the device is on", key="fbar"):
        tb = T[T.run_min > 0].sort_values("avg_run", ascending=False); f = go.Figure()
        for cl in CLASS_ORDER:
            sub = tb[tb["class"] == cl]
            f.add_bar(x=[f"D{int(i)}" for i in sub.DeviceID], y=sub.avg_run, name=cl, marker_color=CLASS_COL[cl], customdata=np.c_[sub.uptime, sub.peak], hovertemplate="%{x}<br><b>%{y:,.0f} kW</b> avg running<br>on %{customdata[0]:.1f} % · peak %{customdata[1]:,.0f} kW<extra></extra>")
        f.update_xaxes(type="category", categoryorder="array", categoryarray=[f"D{int(i)}" for i in tb.DeviceID]); f.update_layout(hovermode="closest", yaxis_title="kW"); show(style(f, 380, True, top=60))
    with b, panel("Uptime vs load level", "bubble size = peak kW · top-right = big and always on", key="fscat"):
        f = go.Figure()
        for cl in CLASS_ORDER:
            sub = T[(T["class"] == cl) & (T.run_min > 0)]
            f.add_scatter(x=sub.uptime, y=sub.avg_run, mode="markers+text", text=[f"{int(i)}" for i in sub.DeviceID], textposition="top center", textfont=dict(size=10, color=MUTED), name=cl,
                          marker=dict(color=CLASS_COL[cl], opacity=.8, size=np.clip(sub.peak.fillna(1) ** .5 * 1.6, 8, 42), line=dict(color="#fff", width=1)),
                          hovertemplate="Device %{text}<br>on %{x:.1f} %<br>avg %{y:,.0f} kW<extra></extra>")
        f.update_layout(hovermode="closest", xaxis_title="on % of measured minutes", yaxis_title="avg kW while running"); show(style(f, 380, True, top=60))

    with panel("When did each device run?", "hours of running per day (KW above the threshold) · grey = idle all day · white = no data", key="fdaily"):
        dm = F["daily"][T.sort_values("avg_run", ascending=False, na_position="last").DeviceID.map(lambda i: f"Device {int(i)}")]
        f = go.Figure(go.Heatmap(z=dm.T.values, x=[f"{d:%d %b}" for d in dm.index], y=list(dm.columns), zmin=0, zmax=24, xgap=1, ygap=1, colorscale=[[0, "#EBEEF5"], [.001, "#FDE9BF"], [.5, "#F59E0B"], [1, "#B45309"]],
                                 colorbar=dict(title="h/day", thickness=10), hovertemplate="%{y} · %{x}<br><b>%{z:.1f} h running</b><extra></extra>"))
        f.update_yaxes(autorange="reversed", type="category"); f.update_xaxes(type="category", dtick=max(1, len(dm) // 18)); f.update_layout(hovermode="closest"); show(style(f, max(320, 19 * len(dm.columns) + 70), False, top=20))
    with panel("At what time of day does each device run?", "share of minutes at that hour with KW above the threshold, over all days", key="fhour"):
        hm = F["hrun"].loc[[f"Device {int(i)}" for i in T.sort_values("avg_run", ascending=False, na_position="last").DeviceID]]
        f = go.Figure(go.Heatmap(z=hm.values, x=[f"{h:02d}:00" for h in hm.columns], y=list(hm.index), zmin=0, zmax=100, xgap=2, ygap=1, colorscale=[[0, "#EBEEF5"], [.01, "#FDE9BF"], [.5, "#F59E0B"], [1, "#B45309"]],
                                 colorbar=dict(title="% on", thickness=10), hovertemplate="%{y} · %{x}<br><b>%{z:.0f} % of the time on</b><extra></extra>"))
        f.update_yaxes(autorange="reversed", type="category"); f.update_layout(hovermode="closest"); show(style(f, max(320, 19 * len(hm) + 70), False, top=20))
    with panel("Plant total load", "hourly average kW, stacked by device (largest 8 + others)", key="ftotal"):
        hr = F["hourly"]; order = hr.mean().sort_values(ascending=False).index; top8 = list(order[:8]); f = go.Figure()
        pal = ["#F59E0B", "#6366F1", "#10B981", "#F43F5E", "#06B6D4", "#8B5CF6", "#3B82F6", "#84CC16"]
        for c, col in zip(top8, pal): f.add_scatter(x=hr.index, y=hr[c], name=c, stackgroup="s", mode="lines", line=dict(width=.6, color=col), fillcolor=col + "AA")
        if len(order) > 8: f.add_scatter(x=hr.index, y=hr[order[8:]].sum(axis=1, min_count=1), name="Others", stackgroup="s", mode="lines", line=dict(width=.6, color="#94A3B8"), fillcolor="#CBD5E1AA")
        f.update_layout(yaxis_title="kW"); f.update_xaxes(rangeslider_visible=True, rangeslider_thickness=.06); show(style(f, 400, True, top=70))

# ================================================================== DEVICE DEEP-DIVE
with tab_dev:
    default = int(T.sort_values("kwh", ascending=False).DeviceID.iloc[0])
    dev = st.selectbox("Device", list(LBL), index=list(LBL).index(default), format_func=lambda i: LBL[i], key="dev_sel")
    A = analyse(int(dev), float(thr)); S = A["stats"]; sess, idl, gap = A["sessions"], A["idles"], A["nodata"]; hour, daily = A["hour"], A["daily"]
    k = st.columns(6)
    with k[0]: kpi("On-time", f"{S['uptime']:.1f} %", f"{fmt_h(S['run_min'])} on · {fmt_h(S['idle_min'])} idle", "info", "⏱️", daily.run_h.values, FORECAST)
    with k[1]: kpi("Energy", f"{S['kwh']:,.0f} kWh", f"{S['days']:.0f} days · {pd.Timestamp(S['first']):%d %b} → {pd.Timestamp(S['last']):%d %b}", "", "🔋", daily.kwh.values, LOAD)
    with k[2]: kpi("Avg load · running", "–" if pd.isna(S["avg_run"]) else f"{S['avg_run']:,.1f} kW", "–" if pd.isna(S["median_run"]) else f"median {S['median_run']:,.1f} · P95 {S['p95_run']:,.1f}", "", "⚙️", daily.avg_run.values, CYAN)
    with k[3]: kpi("Peak", "–" if pd.isna(S["peak"]) else f"{S['peak']:,.1f} kW", "highest 1-minute reading", "", "📈")
    with k[4]: kpi("Sessions", f"{S['sessions']:,}", f"{S['sessions_per_day']:.1f} per day · median {fmt_h(S['median_session_min']) if pd.notna(S['median_session_min']) else '–'}", "", "🔁")
    with k[5]: kpi("No data", f"{100 - S['coverage']:.1f} %", f"{fmt_h(S['nodata_min'])} missing · longest gap {fmt_h(S['longest_gap_min'])}", "warn" if S["coverage"] < 95 else "good", "🕳️")
    st.write("")

    # ---- plain-language findings
    pts = []
    pts.append(f"<b>{S['class']}</b> - KW is above {thr:g} kW in <b>{S['uptime']:.1f} %</b> of the measured minutes ({fmt_h(S['run_min'])} of {fmt_h(S['run_min'] + S['idle_min'])}); the rest is idle, and {100 - S['coverage']:.1f} % of the period has no data at all.")
    if S["run_min"]:
        act = [h for h in range(24) if pd.notna(hour.run_pct[h]) and hour.run_pct[h] >= max(10, .5 * hour.run_pct.max())]
        pts.append(f"Most active hours: <b>{hour_ranges(act)}</b> (busiest hour {int(hour.run_pct.idxmax()):02d}:00, on {hour.run_pct.max():.0f} % of the time; quietest {int(hour.run_pct.idxmin()):02d}:00, on {hour.run_pct.min():.0f} %).")
        if S["uptime"] < 99.5 and len(sess):
            pts.append(f"<b>{S['sessions']:,}</b> running sessions · median {fmt_h(S['median_session_min'])}, longest {fmt_h(S['longest_session_min'])} · typical start hour <b>{int(sess.start_hour.mode().iloc[0]):02d}:00</b> · longest idle spell {fmt_h(S['longest_idle_min'])}.")
        if (daily.weekend.any() and (~daily.weekend).any()) and (S["wd_run_h"] + S["we_run_h"]) > 0:
            wd, we = daily[~daily.weekend].run_h.mean(), daily[daily.weekend].run_h.mean()
            pts.append(f"Weekdays run <b>{wd:.1f} h/day</b>, weekends <b>{we:.1f} h/day</b>" + (" - clearly lower at weekends." if we < .7 * wd else " - no strong weekday/weekend difference." if abs(we - wd) < .15 * max(wd, 1e-9) else "."))
        pts.append(f"While running the load is typically <b>{S['median_run']:,.1f} kW</b> (P95 {S['p95_run']:,.1f}, peak {S['peak']:,.1f}); minute-to-minute it moves about <b>{S['step_pct']:.1f} %</b> of that level." if pd.notna(S["step_pct"]) else "")
        pts.append(f"First time above threshold: {pd.Timestamp(S['first_run']):%d %b %H:%M} · last time: <b>{pd.Timestamp(S['last_run']):%d %b %H:%M}</b>" + (" (stopped before the data ends)." if pd.Timestamp(S["last"]) - pd.Timestamp(S["last_run"]) > pd.Timedelta(hours=6) else "."))
    r24 = "–" if pd.isna(S["r2_24h"]) else f"{S['r2_24h']:.2f}"
    pts.append(f"<b>Forecast hint:</b> {S['verdict']} <span style='color:#64748B'>(persistence R² at 30 min = {S['r2_30']:.2f}, same-time-yesterday R² = {r24})</span>")
    note("<ul style='margin:.1rem 0 0 1rem;padding:0'>" + "".join(f"<li style='margin:.25rem 0'>{p}</li>" for p in pts if p) + "</ul>")

    sub_tl, sub_when, sub_sess, sub_dist, sub_pred = st.tabs(["Timeline", "When it runs", "Sessions & gaps", "Distribution", "Predictability"])
    D = load_device(int(dev))

    # ---------------- timeline
    with sub_tl:
        c = st.columns([1.3, 1.3, 3])
        sig = c[0].selectbox("Signal", ["Total load (kW)", "Phase load R / Y / B (kW)", "Phase current (A)", "Voltage (V)", "Power factor", "Apparent power (kVA)"], key="sig")
        res = c[1].selectbox("Resolution", ["1 min (raw)", "10 min", "1 hour"], index=1, key="res")
        cols = {"Total load (kW)": ["KW"], "Phase load R / Y / B (kW)": PHASES, "Phase current (A)": ["I1", "I2", "I3"], "Voltage (V)": ["V1", "V2", "V3"], "Power factor": ["PF"], "Apparent power (kVA)": ["KVA"]}[sig]
        view = D[cols] if res.startswith("1 min") else D[cols].resample("10min" if res.startswith("10") else "1h").mean()
        pal = [LOAD] if len(cols) == 1 else [PHASE_COL["RKW"], PHASE_COL["YKW"], PHASE_COL["BKW"]]; f = go.Figure()
        for cn, col in zip(cols, pal):
            f.add_trace(go.Scattergl(x=view.index, y=view[cn], name=cn, mode="lines", line=dict(color=col, width=1.3), connectgaps=False))
        if cols == ["KW"]: hline(f, thr, f"threshold {thr:g} kW")
        f.update_xaxes(rangeslider_visible=True, rangeslider_thickness=.07); show(style(f, 400, len(cols) > 1, top=40))
        st.caption("Lines break wherever minutes are missing (no data). Zero is real idle time.")

        with panel("Day × hour calendar", "each cell = one hour of one day", key="cal"):
            met = st.radio("Show", ["Average kW", "% of the hour KW is there", "% of the hour with no data"], horizontal=True, key="calmet")
            z = {"Average kW": A["cal"]["kw"], "% of the hour KW is there": A["cal"]["run"], "% of the hour with no data": A["cal"]["miss"]}[met]
            zmax = float(np.nanpercentile(z.values, 98)) if met == "Average kW" and np.isfinite(z.values).any() else 100
            cs_ = [[0, "#EBEEF5"], [.02, "#FDE9BF"], [.5, "#F59E0B"], [1, "#B45309"]] if met != "% of the hour with no data" else [[0, "#FFFFFF"], [.01, "#FFE4E8"], [1, "#BE123C"]]
            f = go.Figure(go.Heatmap(z=z.values, x=[f"{h:02d}:00" for h in z.columns], y=[f"{d:%d %b %a}" for d in z.index], zmin=0, zmax=max(zmax, 1e-6), xgap=2, ygap=1, colorscale=cs_,
                                     colorbar=dict(thickness=10, len=.6), hovertemplate="%{y} · %{x}<br><b>%{z:.1f}</b><extra></extra>"))
            f.update_yaxes(autorange="reversed", type="category", dtick=max(1, len(z) // 30)); f.update_layout(hovermode="closest"); show(style(f, min(1100, max(360, 15 * len(z) + 60)), False, top=20))

    # ---------------- when it runs
    with sub_when:
        a, b = st.columns([1.4, 1])
        with a, panel("Typical day", "bars = % of the time on at that hour · line = kW when running (band = P10–P90)", key="hourprof"):
            f = make_subplots(specs=[[{"secondary_y": True}]]); hx = [f"{h:02d}:00" for h in range(24)]
            f.add_bar(x=hx, y=hour.run_pct, name="% of time on", marker_color="#FDE9BF", secondary_y=False)
            f.add_scatter(x=hx, y=hour.p90, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip", secondary_y=True)
            f.add_scatter(x=hx, y=hour.p10, mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(99,102,241,.14)", name="P10–P90 while running", hoverinfo="skip", secondary_y=True)
            f.add_scatter(x=hx, y=hour.p50, mode="lines+markers", line=dict(color=FORECAST, width=2.5), marker=dict(size=5), name="Median kW while running", secondary_y=True)
            f.update_yaxes(title_text="% of time on", range=[0, 100], secondary_y=False); f.update_yaxes(title_text="kW", rangemode="tozero", secondary_y=True, showgrid=False); show(style(f, 380, True, top=70))
        with b, panel("Weekday × hour", "% of minutes with KW above the threshold", key="dowh"):
            dh = A["dow_hour"]
            f = go.Figure(go.Heatmap(z=dh.values, x=[f"{h:02d}" for h in dh.columns], y=DOW, zmin=0, zmax=100, xgap=2, ygap=2, colorscale=[[0, "#EBEEF5"], [.01, "#FDE9BF"], [.5, "#F59E0B"], [1, "#B45309"]],
                                     colorbar=dict(title="% on", thickness=10), hovertemplate="%{y} %{x}:00<br><b>%{z:.0f} % on</b><extra></extra>"))
            f.update_yaxes(autorange="reversed"); f.update_layout(hovermode="closest"); show(style(f, 380, False, top=20))
        with panel("Running hours per day", "orange = weekday · blue = weekend · hover for energy and average load", key="drun"):
            f = go.Figure(go.Bar(x=daily.index, y=daily.run_h, marker_color=[BLUE if w else LOAD for w in daily.weekend], customdata=np.c_[daily.kwh, daily.avg_run, daily.missing_h],
                                 hovertemplate="%{x|%a %d %b}<br><b>%{y:.1f} h running</b><br>%{customdata[0]:,.0f} kWh · avg %{customdata[1]:.0f} kW<br>%{customdata[2]:.1f} h no data<extra></extra>"))
            f.update_layout(hovermode="closest", yaxis_title="hours"); show(style(f, 300, False, top=20))

    # ---------------- sessions & gaps
    with sub_sess:
        if not len(sess): st.info("This device never goes above the threshold.")
        else:
            a, b, c = st.columns(3)
            with a, panel("Session length", "how long each running stretch lasts", key="sdur"):
                f = go.Figure(go.Histogram(x=sess.minutes / 60, nbinsx=40, marker_color=LOAD)); f.update_layout(hovermode="closest", xaxis_title="hours", yaxis_title="sessions"); show(style(f, 300, False, top=20))
            with b, panel("Idle spell length", "how long it stays off between sessions", key="idur"):
                f = go.Figure(go.Histogram(x=idl.minutes / 60, nbinsx=40, marker_color="#94A3B8")) if len(idl) else go.Figure(); f.update_layout(hovermode="closest", xaxis_title="hours", yaxis_title="idle spells"); show(style(f, 300, False, top=20))
            with c, panel("Session start hour", "when machines are switched on", key="sstart"):
                cnt = sess.start_hour.value_counts().reindex(range(24), fill_value=0); f = go.Figure(go.Bar(x=[f"{h:02d}" for h in cnt.index], y=cnt.values, marker_color=FORECAST)); f.update_layout(hovermode="closest", yaxis_title="sessions"); show(style(f, 300, False, top=20))
            with panel("Sessions", "newest first · avg/peak kW and energy per session", key="stable"):
                st.dataframe(sess.sort_values("start", ascending=False).assign(minutes=lambda x: x.minutes.astype(int))[["start", "end", "minutes", "avg_kw", "peak_kw", "kwh"]].rename(columns={"minutes": "Minutes", "avg_kw": "Avg kW", "peak_kw": "Peak kW", "kwh": "kWh", "start": "Start", "end": "End"}),
                             hide_index=True, width="stretch", height=300, column_config={"Avg kW": st.column_config.NumberColumn(format="%.1f"), "Peak kW": st.column_config.NumberColumn(format="%.1f"), "kWh": st.column_config.NumberColumn(format="%.1f")})
                st.download_button("⬇ Sessions (CSV)", sess.to_csv(index=False).encode(), f"device_{dev}_sessions.csv", "text/csv", on_click="ignore")
        with panel("Missing-data gaps", "stretches where the meter sent nothing (longest first)", key="gaps"):
            if not len(gap): st.caption("No missing minutes - the device reported every minute.")
            else: st.dataframe(gap.sort_values("minutes", ascending=False).head(100)[["start", "end", "minutes"]].rename(columns={"start": "From", "end": "To", "minutes": "Minutes missing"}), hide_index=True, width="stretch", height=240)

    # ---------------- distribution
    with sub_dist:
        on = D["KW"] > thr; kr = D.loc[on, "KW"].dropna()
        if kr.empty: st.info("This device never goes above the threshold.")
        else:
            a, b = st.columns([1.3, 1])
            with a, panel("Load distribution while running", "how often each kW level occurs (KW above the threshold)", key="hist"):
                f = go.Figure(go.Histogram(x=kr, nbinsx=60, marker_color=LOAD, hovertemplate="%{x} kW<br>%{y:,} minutes<extra></extra>"))
                for q, col in ((.5, FORECAST), (.95, BAD)): f.add_vline(x=float(kr.quantile(q)), line_dash="dot", line_color=col, annotation_text=f"P{int(q * 100)} {kr.quantile(q):,.1f}", annotation_font=dict(size=11, color=col))
                f.update_layout(hovermode="closest", xaxis_title="kW", yaxis_title="minutes"); show(style(f, 340, False, top=30))
            with b, panel("Percentiles", "kW while running", key="pct"):
                qs = [.01, .05, .25, .5, .75, .95, .99, 1.0]
                table(pd.DataFrame({"Percentile": [f"P{int(q * 100)}" if q < 1 else "Max" for q in qs], "kW": [kr.quantile(q) for q in qs]}), 1)
                st.caption(f"Mean {kr.mean():,.1f} kW · standard deviation {kr.std():,.1f} kW · coefficient of variation {kr.std() / kr.mean() * 100:.0f} %")
            with panel("kW by hour of day", "spread of the load while running - box = middle 50 %, line = median", key="box"):
                f = go.Figure(); smp = kr.groupby(kr.index.hour, group_keys=False).apply(lambda s_: s_.sample(min(len(s_), 1500), random_state=1))
                for h in range(24):
                    v_ = smp[smp.index.hour == h]
                    if len(v_): f.add_box(y=v_.values, name=f"{h:02d}", marker_color=LOAD, line=dict(width=1.2), boxpoints=False, showlegend=False)
                f.update_layout(hovermode="closest", yaxis_title="kW", xaxis_title="hour of day"); show(style(f, 340, False, top=20))
            ph = D.loc[on, PHASES].dropna()
            if len(ph):
                a, b, c = st.columns(3)
                with a, panel("Phase share", "average kW per phase while running", key="phase"):
                    f = go.Figure(go.Bar(x=["R", "Y", "B"], y=ph.mean().values, marker_color=[PHASE_COL[p] for p in PHASES], text=[f"{v:,.1f}" for v in ph.mean().values], textposition="outside")); f.update_layout(hovermode="closest", yaxis_title="kW"); show(style(f, 300, False, top=20))
                with b, panel("Phase imbalance per day", "(max − min) ÷ mean of R/Y/B, daily median", key="imb"):
                    imb = ((ph.max(axis=1) - ph.min(axis=1)) / ph.mean(axis=1).replace(0, np.nan) * 100).groupby(ph.index.normalize()).median()
                    f = go.Figure(go.Scatter(x=imb.index, y=imb.values, mode="lines+markers", line=dict(color=VIOLET, width=2), marker=dict(size=5))); f.update_layout(yaxis_title="%"); show(style(f, 300, False, top=20))
                with c, panel("Power factor", "while running", key="pf"):
                    pf = D.loc[on, "PF"].dropna()
                    f = go.Figure(go.Histogram(x=pf, nbinsx=40, marker_color=CYAN)); f.update_layout(hovermode="closest", xaxis_title="PF", yaxis_title="minutes"); show(style(f, 300, False, top=20))
            vi = D.loc[on, ["V1", "V2", "V3", "I1", "I2", "I3", "F"]].mean()
            note(f"While running - average voltage <b>{vi.V1:,.0f} / {vi.V2:,.0f} / {vi.V3:,.0f} V</b> · average current <b>{vi.I1:,.0f} / {vi.I2:,.0f} / {vi.I3:,.0f} A</b> · frequency <b>{vi.F:.2f} Hz</b>.")

    # ---------------- predictability
    with sub_pred:
        sk = A["skill"]; ST = stationarity(int(dev))
        a, b = st.columns(2)
        with a, panel("How predictable is the next value?", "R² of the naive forecast 'KW now = KW at that time earlier' · higher is easier to model", key="skill"):
            f = go.Figure(go.Bar(x=sk.horizon, y=sk.r2, marker_color=[GOOD if v >= .8 else LOAD if v >= .4 else BAD for v in sk.r2.fillna(0)], text=[f"{v:.2f}" if pd.notna(v) else "–" for v in sk.r2], textposition="outside",
                                 customdata=sk.mae, hovertemplate="look-back %{x}<br>R² %{y:.2f}<br>MAE %{customdata:.1f} kW<extra></extra>"))
            f.update_layout(hovermode="closest", yaxis_title="R²", xaxis_title="how far back the naive forecast looks"); f.update_yaxes(range=[min(-.1, float(sk.r2.min(skipna=True) if sk.r2.notna().any() else 0) - .1), 1.1]); show(style(f, 360, False, top=20))
            st.caption("Short look-backs ≈ how smooth the signal is. '24 h' ≈ strength of a daily routine, '7 days' ≈ weekly routine. Negative R² = worse than always guessing the average.")
        with b, panel("Autocorrelation", "10-minute average KW · peaks at 24 h mean a daily cycle", key="acf"):
            if ST["acf"] is None: st.info("Not enough varying data for an autocorrelation plot.")
            else:
                ac = ST["acf"]; lag_h = np.arange(len(ac)) * 10 / 60; f = go.Figure(go.Scatter(x=lag_h, y=ac, mode="lines", line=dict(color=FORECAST, width=1.8), fill="tozeroy", fillcolor="rgba(99,102,241,.12)"))
                bound = 1.96 / np.sqrt(ST["n"]); f.add_hrect(y0=-bound, y1=bound, fillcolor="#94A3B8", opacity=.2, line_width=0)
                for dd in range(1, int(lag_h.max() // 24) + 1): f.add_vline(x=24 * dd, line_dash="dot", line_color="#CBD5E1")
                f.update_layout(xaxis_title="lag (hours)", yaxis_title="correlation"); f.update_yaxes(range=[min(-.3, float(ac.min()) - .05), 1.02]); show(style(f, 360, False, top=20))
                dayp = ac[144] if len(ac) > 144 else np.nan
                st.caption(f"Correlation with the same time yesterday: {'–' if pd.isna(dayp) else f'{dayp:.2f}'} · grey band = what pure noise would give.")
        p = ST["adf_p"]
        if pd.notna(p):
            note(f"<b>Stationarity (ADF test on 10-min means):</b> p = {p:.4f} → " + ("the series is <b>stationary</b> (no trend / drift) - ARIMA with d = 0 is reasonable." if p < .05 else "<b>not stationary</b> - there is a level shift or trend; try d = 1 or model the running periods separately.") +
                 " Long idle stretches mixed with running periods often make a series look non-stationary even when each part is stable.", "" if p < .05 else "amber")
        note(f"<b>Forecast hint for Device {dev}:</b> {S['verdict']}. This is a rule of thumb from uptime, session pattern and the scores above - confirm it with your prediction notebook.", "amber")

# ================================================================== COMPARE
with tab_cmp:
    pool = T[T.run_min >= 300].sort_values("avg_run", ascending=False).DeviceID.astype(int).tolist() or T.DeviceID.astype(int).tolist()
    sel = st.multiselect("Devices to compare (up to 8)", list(LBL), default=pool[:4], format_func=lambda i: LBL[i], max_selections=8, key="cmp")
    if len(sel) < 1: st.info("Pick at least one device.")
    else:
        names = [f"Device {i}" for i in sel]; pal = ["#F59E0B", "#6366F1", "#10B981", "#F43F5E", "#06B6D4", "#8B5CF6", "#3B82F6", "#84CC16"]
        mode = st.radio("Daily profile shows", ["Average kW (all minutes)", "% of the time on", "Average kW as % of own busiest hour"], horizontal=True, key="cmpmode")
        prof = (F["hkw"] if mode != "% of the time on" else F["hrun"]).loc[names]
        if mode.endswith("busiest hour"): prof = prof.div(prof.max(axis=1).replace(0, np.nan), axis=0) * 100
        a, b = st.columns(2)
        with a, panel("Typical day", "by hour of day, averaged over all days", key="cprof"):
            f = go.Figure()
            for nme, col in zip(names, pal): f.add_scatter(x=[f"{h:02d}:00" for h in prof.columns], y=prof.loc[nme], name=nme, mode="lines+markers", line=dict(color=col, width=2.2), marker=dict(size=4))
            f.update_layout(yaxis_title="%" if mode != "Average kW (all minutes)" else "kW"); show(style(f, 380, True, top=70))
        with b, panel("Does the load move together?", "correlation of hourly average kW (1 = rise and fall together)", key="ccorr"):
            hrs = F["hourly"][names]; cr = hrs.corr() if len(names) > 1 else None
            if cr is None or cr.isna().all().all(): st.info("Pick two or more devices that vary over time.")
            else:
                f = go.Figure(go.Heatmap(z=cr.values, x=names, y=names, zmin=-1, zmax=1, colorscale=[[0, "#F43F5E"], [.5, "#F8FAFC"], [1, "#4F46E5"]], text=np.round(cr.values, 2), texttemplate="%{text}", colorbar=dict(thickness=10)))
                f.update_yaxes(autorange="reversed"); f.update_layout(hovermode="closest"); show(style(f, 380, False, top=20))
        with panel("Hourly load over time", "average kW per hour · zoom with the slider", key="cover"):
            f = go.Figure()
            for nme, col in zip(names, pal): f.add_trace(go.Scattergl(x=F["hourly"].index, y=F["hourly"][nme], name=nme, mode="lines", line=dict(color=col, width=1.4), connectgaps=False))
            f.update_layout(yaxis_title="kW"); f.update_xaxes(rangeslider_visible=True, rangeslider_thickness=.06); show(style(f, 400, True, top=70))
        with panel("Side by side", "", key="ccmp"):
            cm = T[T.DeviceID.isin(sel)].set_index("DeviceID").loc[sel]
            table(pd.DataFrame({"Device": [f"{i}" for i in sel], "On %": cm.uptime.values, "Avg kW (on)": cm.avg_run.values, "Peak kW": cm.peak.values, "kWh": cm.kwh.values, "Sessions/day": cm.sessions_per_day.values,
                                "R² 30 min": cm.r2_30.values, "R² 24 h": cm.r2_24h.values}), 2)
