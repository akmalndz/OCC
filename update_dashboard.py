"""
SPY Options Command Center — Day Trade Edition
Fetches live SPY data, computes signals, writes index.html
Architecture mirrors NVDA OCC v6 — same GitHub Actions workflow
"""

import yfinance as yf
import datetime, pytz, math, json
import feedparser
import numpy as np
from pathlib import Path

CT = pytz.timezone("America/Chicago")
ET = pytz.timezone("America/New_York")

# ─── BRANDO / @EliteOptions2 SPX→SPY LEVELS (SPX÷10.18) ──────────────────────
# Source: TrendSpider @EliteOptions2, SPYOptions Substack
LEVELS = [
    (702.4,  "MO", "Monthly Wall $7T",        True ),
    (688.0,  "MO", "Monthly R1",              False),
    (683.1,  "MO", "Monthly S/R",             False),
    (677.9,  "WK", "Weekly Long Target",       True ),
    (673.4,  "DY", "Bull Entry Zone",          True ),
    (671.0,  "DY", "Key Pivot (Yellow Line)",  True ),
    (668.0,  "DY", "Bear Entry Zone",          True ),
    (664.4,  "WK", "Weekly Short Target",      True ),
    (660.1,  "DY", "Critical Support",         True ),
    (658.2,  "DY", "Daily S1",                 False),
    (655.0,  "WK", "Weekly S1",                False),
    (651.6,  "MO", "Monthly Demand",           True ),
    (640.6,  "MO", "Major Monthly Floor",      True ),
    (638.6,  "MO", "Monthly Floor",            True ),
]

TF_COLOR = {"MO": "#94a3b8", "WK": "#60a5fa", "DY": "#94a3b8"}

# ─── SESSIONS (CT) ─────────────────────────────────────────────────────────────
SESSIONS = [
    {"name": "PRE-MKT",    "start": (8, 0),  "end": (9, 15), "color": "#475569", "trade": False, "advice": "Bias formation only. No entries."},
    {"name": "FAKE-OUT",   "start": (9, 15), "end": (9, 45), "color": "#f87171", "trade": False, "advice": "ORB forming. FAKE-OUT ZONE — wait."},
    {"name": "ORB WINDOW", "start": (9, 45), "end": (10, 30),"color": "#4ade80", "trade": True,  "advice": "PRIMARY WINDOW. KAMA cross + Ichimoku confirm."},
    {"name": "HARD CUTOFF","start": (10, 30),"end": (10, 31),"color": "#f87171", "trade": False, "advice": "10:30 HARD EXIT. Close all positions NOW."},
    {"name": "DEAD ZONE",  "start": (10, 31),"end": (23, 59),"color": "#475569", "trade": False, "advice": "No new entries after 10:30 CT."},
]

def get_session(ct_now):
    t = ct_now.hour * 60 + ct_now.minute
    for s in SESSIONS:
        if s["start"][0]*60+s["start"][1] <= t < s["end"][0]*60+s["end"][1]:
            return s
    return SESSIONS[-1]

# ─── MATH ──────────────────────────────────────────────────────────────────────
def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def rsi(series, period=14):
    d = series.diff()
    g = d.clip(lower=0).rolling(period).mean()
    l = (-d.clip(upper=0)).rolling(period).mean()
    return (100 - (100 / (1 + g / l.replace(0, np.nan)))).fillna(50)

def atr(hist, period=14):
    h, l, c = hist["High"], hist["Low"], hist["Close"]
    tr = np.maximum(h - l, np.maximum(abs(h - c.shift(1)), abs(l - c.shift(1))))
    return tr.rolling(period).mean()

def macd_histogram(series, fast=12, slow=26, signal=9):
    m = ema(series, fast) - ema(series, slow)
    return m - ema(m, signal)

def vwap_calc(hist):
    hlc3 = (hist["High"] + hist["Low"] + hist["Close"]) / 3
    return (hlc3 * hist["Volume"]).cumsum() / hist["Volume"].cumsum()

def kama_calc(hist, length=10, fast_sc=2, slow_sc=30):
    hlc3 = (hist["High"] + hist["Low"] + hist["Close"]) / 3
    ch = abs(hlc3 - hlc3.shift(length))
    vol = abs(hlc3 - hlc3.shift(1)).rolling(length).sum()
    er = (ch / vol.replace(0, np.nan)).fillna(0)
    fc = 2. / (fast_sc + 1); sc_ = 2. / (slow_sc + 1)
    sc = (er * (fc - sc_) + sc_) ** 2
    kv = np.zeros(len(hlc3)); ha = hlc3.values; sa = sc.values; kv[0] = ha[0]
    for i in range(1, len(ha)):
        kv[i] = kv[i-1] if (np.isnan(sa[i]) or np.isnan(ha[i])) else kv[i-1] + sa[i] * (ha[i] - kv[i-1])
    return kv, float(er.iloc[-1])

def efficiency_ratio(series, period=20):
    if len(series) < period + 1: return 0.5
    direction = abs(series.iloc[-1] - series.iloc[-1-period])
    noise = abs(series.diff()).iloc[-period:].sum()
    return round(float(direction / noise) if noise > 0 else 0, 3)

def adx_calc(hist, period=14):
    h, l, c = hist["High"], hist["Low"], hist["Close"]
    plus_dm = (h.diff()).clip(lower=0)
    minus_dm = (-l.diff()).clip(lower=0)
    mask = plus_dm < minus_dm; plus_dm[mask] = 0
    mask2 = minus_dm <= plus_dm; minus_dm[mask2] = 0
    tr = np.maximum(h - l, np.maximum(abs(h - c.shift(1)), abs(l - c.shift(1))))
    tr14 = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / tr14.replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(period).mean() / tr14.replace(0, np.nan))
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
    return round(float(dx.rolling(period).mean().iloc[-1]), 1)

def rvol_calc(hist, period=5, baseline=20):
    recent = hist["Volume"].iloc[-period:].mean()
    base = hist["Volume"].iloc[-baseline-period:-period].mean()
    return round(float(recent / base) if base > 0 else 1.0, 2)

# ─── ICHIMOKU ──────────────────────────────────────────────────────────────────
def calc_ichimoku(hist):
    if len(hist) < 60:
        return {"bull": False, "bear": False, "label": "N/A", "detail": "No data", "cloud_top": None, "cloud_bottom": None, "cloud_bull": None}
    h, l, c = hist["High"], hist["Low"], hist["Close"]
    tenkan = (h.rolling(9).max() + l.rolling(9).min()) / 2
    kijun  = (h.rolling(26).max() + l.rolling(26).min()) / 2
    spanA  = (tenkan + kijun) / 2
    spanB  = (h.rolling(52).max() + l.rolling(52).min()) / 2
    tn = float(tenkan.iloc[-1]); kn = float(kijun.iloc[-1])
    an = float(spanA.iloc[-1]);  bn = float(spanB.iloc[-1])
    pn = float(c.iloc[-1])
    ct = max(an, bn); cb = min(an, bn); cbull = an > bn
    above = pn > ct; below = pn < cb
    bull = above and tn > kn
    bear = below and tn < kn
    if bull:    lbl = "BULL"; det = f"Above cloud · T>{kn:.2f}"
    elif bear:  lbl = "BEAR"; det = f"Below cloud · T<{kn:.2f}"
    elif above: lbl = "WEAK BULL"; det = "Above cloud, T≤K"
    elif below: lbl = "WEAK BEAR"; det = "Below cloud, T≥K"
    else:       lbl = "IN CLOUD"; det = f"${cb:.2f}–${ct:.2f}"
    return {"bull": bull, "bear": bear, "label": lbl, "detail": det,
            "cloud_top": round(ct, 2), "cloud_bottom": round(cb, 2), "cloud_bull": cbull,
            "tenkan": round(tn, 2), "kijun": round(kn, 2)}

# ─── KAMA SIGNAL ───────────────────────────────────────────────────────────────
def calc_kama_signal(hist):
    if len(hist) < 25:
        return {"bull": False, "bear": False, "label": "N/A", "detail": "No data", "kama_val": None, "er": None}
    kv, er_val = kama_calc(hist)
    kn = kv[-1]; kp = kv[-2]
    hlc3_n = float((hist["High"].iloc[-1] + hist["Low"].iloc[-1] + hist["Close"].iloc[-1]) / 3)
    rising = kn > kp; falling = kn < kp
    bull = hlc3_n > kn and rising
    bear = hlc3_n < kn and falling
    if bull:       lbl = "BULL"; det = f"HLC3 {hlc3_n:.2f} > KAMA {kn:.2f} · ↑"
    elif bear:     lbl = "BEAR"; det = f"HLC3 {hlc3_n:.2f} < KAMA {kn:.2f} · ↓"
    elif hlc3_n > kn: lbl = "WEAK BULL"; det = f"Above KAMA {kn:.2f}, not rising"
    elif hlc3_n < kn: lbl = "WEAK BEAR"; det = f"Below KAMA {kn:.2f}, not falling"
    else:          lbl = "NEUTRAL"; det = "On KAMA"
    return {"bull": bull, "bear": bear, "label": lbl, "detail": det,
            "kama_val": round(kn, 2), "er": er_val, "hlc3": round(hlc3_n, 2)}

# ─── RSI SIGNAL ────────────────────────────────────────────────────────────────
def calc_rsi_signal(hist, period=14):
    if len(hist) < period + 5:
        return {"bull": False, "bear": False, "label": "N/A", "detail": "No data", "rsi_val": None}
    rs = rsi(hist["Close"], period)
    rn = float(rs.iloc[-1]); rp = float(rs.iloc[-2])
    cross_up   = rp < 50 and rn >= 50
    cross_down = rp > 50 and rn <= 50
    bull = cross_up  or (rn < 35 and rn > rp)   # cross above 50 or OS bounce
    bear = cross_down or (rn > 65 and rn < rp)  # cross below 50 or OB rejection
    zone = "OB" if rn > 70 else ("OS" if rn < 30 else "MID")
    if bull:   lbl = "BULL"; det = f"RSI {rp:.0f}→{rn:.0f} cross↑50"
    elif bear: lbl = "BEAR"; det = f"RSI {rp:.0f}→{rn:.0f} cross↓50"
    else:      lbl = zone; det = f"RSI {rn:.1f} — wait 50 cross"
    return {"bull": bull, "bear": bear, "label": lbl, "detail": det, "rsi_val": round(rn, 1)}

# ─── MACD SIGNAL ───────────────────────────────────────────────────────────────
def calc_macd_signal(hist):
    if len(hist) < 30:
        return {"bull": False, "bear": False, "label": "N/A", "detail": "No data", "macd_h": None}
    mh = macd_histogram(hist["Close"])
    mn = float(mh.iloc[-1]); mp = float(mh.iloc[-2])
    rs = rsi(hist["Close"])
    rn = float(rs.iloc[-1])
    cross_bull = mp < 0 and mn >= 0
    cross_bear = mp > 0 and mn <= 0
    bull = cross_bull and 35 < rn < 70
    bear = cross_bear and 30 < rn < 65
    if bull:   lbl = "BULL"; det = f"MACD cross↑0 · RSI {rn:.0f}"
    elif bear: lbl = "BEAR"; det = f"MACD cross↓0 · RSI {rn:.0f}"
    else:
        lbl = "WAIT"; det = f"MACD {mn:+.4f} · RSI {rn:.0f}"
    return {"bull": bull, "bear": bear, "label": lbl, "detail": det, "macd_h": round(mn, 4)}

# ─── VWAP SIGNAL ───────────────────────────────────────────────────────────────
def calc_vwap_signal(hist):
    if len(hist) < 10:
        return {"bull": False, "bear": False, "label": "N/A", "detail": "No data", "vwap_val": None}
    vw = vwap_calc(hist)
    vn = float(vw.iloc[-1]); price = float(hist["Close"].iloc[-1])
    bull = price > vn; bear = price < vn
    lbl = "ABOVE" if bull else "BELOW"
    det = f"${price:.2f} {'>' if bull else '<'} VWAP ${vn:.2f}"
    return {"bull": bull, "bear": bear, "label": lbl, "detail": det, "vwap_val": round(vn, 2)}

# ─── MTF ROW ───────────────────────────────────────────────────────────────────
def tf_row(hist, label):
    if len(hist) < 30: return None
    c = hist["Close"]
    e8  = round(float(ema(c, 8).iloc[-1]),  2)
    e21 = round(float(ema(c, 21).iloc[-1]), 2)
    price = float(c.iloc[-1])
    rs = rsi(c)
    rn = round(float(rs.iloc[-1]), 1)
    mh = macd_histogram(c)
    mn = round(float(mh.iloc[-1]), 4)
    mp = round(float(mh.iloc[-2]), 4)
    vn_val = hist["Volume"].iloc[-1]
    vm = float(hist["Volume"].rolling(20).mean().iloc[-1]) or 1
    vr = round(vn_val / vm, 2)
    kv, er_v = kama_calc(hist)
    kn = round(kv[-1], 2); kp = round(kv[-2], 2)
    hlc3_n = float((hist["High"].iloc[-1] + hist["Low"].iloc[-1] + hist["Close"].iloc[-1]) / 3)
    bK = hlc3_n > kv[-1]
    bV_vwap = False
    try:
        vwap_v = vwap_calc(hist)
        bV_vwap = price > float(vwap_v.iloc[-1])
    except: pass
    bR = rn > 50; bM = mn > 0; bVol = vr >= 1.0
    score = sum([bK, bR, bM, bV_vwap])
    bias = "BULL" if score >= 3 else ("BEAR" if score <= 1 else "NEUTRAL")
    bc = "#4ade80" if bias == "BULL" else ("#f87171" if bias == "BEAR" else "#fbbf24")
    return {"label": label, "price": round(price, 2), "kama": kn, "rsi": rn, "macd_h": mn,
            "vol_ratio": vr, "bK": bK, "bR": bR, "bM": bM, "bVwap": bV_vwap, "bVol": bVol,
            "score": score, "bias": bias, "bias_color": bc, "er": er_v}

# ─── OPTIONS CHAIN ─────────────────────────────────────────────────────────────
def fetch_options(price):
    try:
        spy = yf.Ticker("SPY")
        exps = spy.options
        if not exps: raise ValueError("No expirations")
        # Find 2 DTE expiry
        today = datetime.date.today()
        target_dte = 2
        best_exp = None; best_diff = 999
        for e in exps:
            exp_date = datetime.datetime.strptime(e, "%Y-%m-%d").date()
            diff = (exp_date - today).days
            if diff >= 1 and abs(diff - target_dte) < best_diff:
                best_diff = abs(diff - target_dte); best_exp = e
        if not best_exp: best_exp = exps[1] if len(exps) > 1 else exps[0]
        dte = (datetime.datetime.strptime(best_exp, "%Y-%m-%d").date() - today).days
        chain = spy.option_chain(best_exp)
        calls_df = chain.calls; puts_df = chain.puts

        def process_chain(df, is_call):
            rows = []
            for _, r in df.iterrows():
                K = float(r["strike"])
                if abs(K - price) > 8: continue
                bid = float(r.get("bid", 0) or 0)
                ask = float(r.get("ask", 0) or 0)
                mid = round((bid + ask) / 2, 2)
                sprd = round(ask - bid, 2)
                vol = int(r.get("volume", 0) or 0)
                oi  = int(r.get("openInterest", 0) or 0)
                iv  = round(float(r.get("impliedVolatility", 0.2) or 0.2) * 100, 1)
                delta = float(r.get("delta", 0.5) or 0.5)
                rows.append({"strike": K, "bid": bid, "ask": ask, "mid": mid,
                             "spread": sprd, "vol": vol, "oi": oi, "iv": iv,
                             "delta": round(abs(delta), 2), "itm": K < price if is_call else K > price})
            return sorted(rows, key=lambda x: abs(x["strike"] - price))[:6]

        calls = process_chain(calls_df, True)
        puts  = process_chain(puts_df,  False)
        # IV of ATM
        atm_call = next((c for c in calls if abs(c["strike"] - price) < 1), None)
        iv_atm = atm_call["iv"] if atm_call else 20.0
        return {"expiry": best_exp, "dte": dte, "calls": calls, "puts": puts, "iv_atm": iv_atm}
    except Exception as e:
        print(f"   Options error: {e}")
        # Synthetic fallback
        strikes = [round(price + d) for d in range(-4, 5)]
        def synth_row(K, is_call):
            m = (price - K) / price
            delta = max(0.05, min(0.95, 0.5 + m * 9))
            cpx = max(0.05, (price - K) * max(delta - 0.5, 0) + 1.2)
            ppx = max(0.05, (K - price) * max(0.5 - delta, 0) + 1.2)
            px = round(cpx if is_call else ppx, 2)
            return {"strike": K, "bid": round(px - 0.05, 2), "ask": round(px + 0.05, 2),
                    "mid": px, "spread": 0.10, "vol": 5000, "oi": 25000,
                    "iv": 18.0, "delta": round(abs(delta), 2), "itm": K < price if is_call else K > price}
        return {"expiry": "synthetic", "dte": 2,
                "calls": [synth_row(K, True)  for K in strikes],
                "puts":  [synth_row(K, False) for K in strikes], "iv_atm": 18.0}

# ─── NEWS ──────────────────────────────────────────────────────────────────────
def fetch_news():
    feeds = [
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=SPY&region=US&lang=en-US",
        "https://www.reutersagency.com/feed/?best-topics=business-finance&post_type=best",
    ]
    articles = []
    for url in feeds:
        try:
            f = feedparser.parse(url)
            for entry in f.entries[:4]:
                title = entry.get("title", "")[:120]
                published = entry.get("published", "")
                # Simple time extraction
                try:
                    import email.utils
                    dt = email.utils.parsedate_to_datetime(published)
                    ct_dt = dt.astimezone(CT)
                    tstr = ct_dt.strftime("%I:%M %p")
                except:
                    tstr = "—"
                articles.append({"title": title, "time": tstr, "src": "MARKET"})
                if len(articles) >= 6: break
        except:
            pass
        if len(articles) >= 6: break
    # Static fallbacks if feeds empty
    if not articles:
        articles = [
            {"title": "Fed speakers on rate path — watch for hawkish tone affecting equities.", "time": "8:15 AM", "src": "FED"},
            {"title": "SPX defended 6754 multiple times. $665.24 breakline key for Monday.", "time": "7:44 AM", "src": "SPY"},
            {"title": "VIX near 23 — elevated. Wider stops warranted. Options premiums inflated.", "time": "7:20 AM", "src": "VIX"},
            {"title": "Futures flat overnight. Dollar slightly lower. No major gap expected.", "time": "6:55 AM", "src": "MACRO"},
        ]
    return articles[:5]

# ─── FETCH ALL SPY DATA ────────────────────────────────────────────────────────
def fetch_all():
    spy = yf.Ticker("SPY")
    info = spy.info
    d1  = spy.history(period="1y",  interval="1d")
    h1  = spy.history(period="30d", interval="1h")
    m30 = spy.history(period="10d", interval="30m")
    m15 = spy.history(period="5d",  interval="15m")
    m5  = spy.history(period="2d",  interval="5m")
    m3  = spy.history(period="1d",  interval="5m")   # yfinance min is 1m or 5m
    m1  = spy.history(period="1d",  interval="1m")

    price     = float(info.get("regularMarketPrice") or info.get("currentPrice") or d1["Close"].iloc[-1])
    prev_close= float(info.get("regularMarketPreviousClose") or info.get("previousClose") or d1["Close"].iloc[-2])
    volume    = int(info.get("regularMarketVolume") or d1["Volume"].iloc[-1])
    avg_vol   = int(info.get("averageVolume") or d1["Volume"].rolling(20).mean().iloc[-1])
    vol_ratio = round(volume / avg_vol, 2) if avg_vol else 1.0

    change    = round(price - prev_close, 2)
    change_pct= round((change / prev_close) * 100, 2)

    week52h   = float(info.get("fiftyTwoWeekHigh")  or d1["High"].max())
    week52l   = float(info.get("fiftyTwoWeekLow")   or d1["Low"].min())
    atr_val   = round(float(atr(d1).iloc[-1]), 2)

    # EMAs on daily
    e50_d = round(float(ema(d1["Close"], 50).iloc[-1]), 2)
    e200_d= round(float(ema(d1["Close"], 200).iloc[-1]), 2)
    above_e50  = price > e50_d
    above_e200 = price > e200_d

    # Signals on different TFs
    sig_kama     = calc_kama_signal(m5)       # Primary: 5m KAMA/HLC3
    sig_ichimoku = calc_ichimoku(h1)          # Context: 1H Ichimoku
    sig_rsi      = calc_rsi_signal(m5)        # 5m RSI
    sig_macd     = calc_macd_signal(m5)       # 5m MACD
    sig_vwap     = calc_vwap_signal(m5)       # 5m VWAP

    # MTF rows
    mtf_rows = {}
    for label, hist in [("1H", h1), ("30m", m30), ("15m", m15), ("5m", m5), ("1m", m1)]:
        r = tf_row(hist, label)
        if r: mtf_rows[label] = r

    # ADX and RVOL on 5m
    adx_val  = adx_calc(m5) if len(m5) > 20 else 20
    rvol_val = rvol_calc(m5) if len(m5) > 25 else 1.0

    # Candles (last 8 x 5m)
    candles = []
    for _, row in m5.tail(8).iterrows():
        ct_t = row.name.astimezone(CT) if hasattr(row.name, 'astimezone') else row.name
        bull_c = float(row["Close"]) >= float(row["Open"])
        candles.append({
            "time": ct_t.strftime("%I:%M %p"),
            "open":  round(float(row["Open"]),  2),
            "high":  round(float(row["High"]),  2),
            "low":   round(float(row["Low"]),   2),
            "close": round(float(row["Close"]), 2),
            "vol":   int(row["Volume"]),
            "bull":  bull_c,
        })

    return {
        "price": round(price, 2), "prev_close": round(prev_close, 2),
        "change": change, "change_pct": change_pct,
        "volume": volume, "avg_vol": avg_vol, "vol_ratio": vol_ratio,
        "week52h": week52h, "week52l": week52l, "atr": atr_val,
        "e50_d": e50_d, "e200_d": e200_d, "above_e50": above_e50, "above_e200": above_e200,
        "sig_kama": sig_kama, "sig_ichimoku": sig_ichimoku,
        "sig_rsi": sig_rsi, "sig_macd": sig_macd, "sig_vwap": sig_vwap,
        "mtf": mtf_rows, "candles": candles,
        "adx": adx_val, "rvol": rvol_val,
        "m5": m5, "h1": h1,
    }

# ─── BUILD SIGNAL ALIGNMENT ────────────────────────────────────────────────────
def build_signals(d, session):
    s_kama = d["sig_kama"]
    s_ichi = d["sig_ichimoku"]
    s_rsi  = d["sig_rsi"]
    s_macd = d["sig_macd"]
    s_vwap = d["sig_vwap"]
    price  = d["price"]
    MAX_SIGNALS = 5

    # Call score
    cc = 0; cr = []
    for s, name in [(s_kama,"KAMA/HLC3"),(s_ichi,"ICHIMOKU"),(s_rsi,"RSI"),(s_macd,"MACD"),(s_vwap,"VWAP")]:
        if s.get("bull"): cc += 1; cr.append(f"+ {name}: {s['detail']}")
        else:             cr.append(f"– {name}: {s['detail']}")

    # Put score
    pc = 0; pr = []
    for s, name in [(s_kama,"KAMA/HLC3"),(s_ichi,"ICHIMOKU"),(s_rsi,"RSI"),(s_macd,"MACD"),(s_vwap,"VWAP")]:
        if s.get("bear"): pc += 1; pr.append(f"+ {name}: {s['detail']}")
        else:             pr.append(f"– {name}: {s['detail']}")

    # Gate bonus checks
    er_ok   = d["sig_kama"].get("er", 0) > 0.35
    adx_ok  = d["adx"] >= 20
    rvol_ok = d["rvol"] >= 1.0
    session_ok = session["trade"]

    gates = {
        "KAMA":    s_kama.get("bull") or s_kama.get("bear"),
        "RSI":     s_rsi.get("bull")  or s_rsi.get("bear"),
        "MACD":    s_macd.get("bull") or s_macd.get("bear"),
        "VWAP":    s_vwap.get("bull") or s_vwap.get("bear"),
        "RVOL":    rvol_ok,
        "ADX":     adx_ok,
        "ER":      er_ok,
        "SESSION": session_ok,
    }

    return {"call_count": cc, "put_count": pc, "max_signals": MAX_SIGNALS,
            "call_reasons": cr, "put_reasons": pr, "gates": gates}

# ─── VERDICT ──────────────────────────────────────────────────────────────────
def get_verdict(al, d, session, opts):
    cc = al["call_count"]; pc = al["put_count"]; ms = al["max_signals"]
    price = d["price"]
    gates = al["gates"]
    gate_pass = sum(1 for v in gates.values() if v)

    # Find best ATM call/put for trade idea
    def best_option(rows, is_call):
        candidates = [r for r in rows if 0.35 <= r["delta"] <= 0.65 and r["vol"] > 100]
        if not candidates: candidates = rows[:3]
        if not candidates: return None
        return sorted(candidates, key=lambda x: abs(x["strike"] - price))[0]

    best_c = best_option(opts.get("calls", []), True)
    best_p = best_option(opts.get("puts",  []), False)

    def idea(opt, cp, stop_pct=0.35, tgt_pct=0.80):
        if not opt: return f"Check chain for {cp} near ATM"
        e = opt["mid"]; st = round(e * (1 - stop_pct), 2); tg = round(e * (1 + tgt_pct), 2)
        dte = opts["dte"]; exp = opts["expiry"]
        return f"${opt['strike']:.0f}{cp[0]} {exp} ({dte}d) ~${e:.2f} | Stop ${st} | Tgt ${tg} | Δ{opt['delta']}"

    if not session["trade"]:
        return {"verdict": f"{session['name']} — NO NEW TRADES", "color": "#475569",
                "bias": "WAIT", "bias_color": "#fbbf24", "explanation": session["advice"],
                "trade_idea": "Next window: ORB 9:45–10:30 AM CT.", "gate_pass": gate_pass}

    cv = "MAX" if cc >= 5 else ("HIGH" if cc >= 4 else "GOOD")
    pv = "MAX" if pc >= 5 else ("HIGH" if pc >= 4 else "GOOD")

    if cc >= 4:
        return {"verdict": f"CALL — {cv} ({cc}/{ms})", "color": "#4ade80",
                "bias": "STRONG BULL", "bias_color": "#4ade80",
                "explanation": f"{cc}/{ms} signals. {gate_pass}/8 gates. 2 DTE | Target +80% | Stop –35%",
                "trade_idea": idea(best_c, "CALL"), "gate_pass": gate_pass}
    elif cc == 3:
        return {"verdict": f"CALL — GOOD ({cc}/{ms})", "color": "#4ade80",
                "bias": "BULLISH", "bias_color": "#4ade80",
                "explanation": f"3/{ms} signals. Confirm KAMA 5m + Ichimoku. 2 DTE.",
                "trade_idea": idea(best_c, "CALL"), "gate_pass": gate_pass}
    elif pc >= 4:
        return {"verdict": f"PUT — {pv} ({pc}/{ms})", "color": "#f87171",
                "bias": "STRONG BEAR", "bias_color": "#f87171",
                "explanation": f"{pc}/{ms} signals. {gate_pass}/8 gates. 2 DTE | Target +80% | Stop –35%",
                "trade_idea": idea(best_p, "PUT"), "gate_pass": gate_pass}
    elif pc == 3:
        return {"verdict": f"PUT — GOOD ({pc}/{ms})", "color": "#f87171",
                "bias": "BEARISH", "bias_color": "#f87171",
                "explanation": f"3/{ms} signals. Volume must confirm — no dead cat bounce.",
                "trade_idea": idea(best_p, "PUT"), "gate_pass": gate_pass}
    elif cc == 2 and cc > pc:
        return {"verdict": f"CALL WATCH ({cc}/{ms})", "color": "#fbbf24",
                "bias": "DEVELOPING", "bias_color": "#fbbf24",
                "explanation": "2 bull signals. Need 3+. Wait for KAMA cross confirmation.",
                "trade_idea": idea(best_c, "CALL"), "gate_pass": gate_pass}
    elif pc == 2 and pc > cc:
        return {"verdict": f"PUT WATCH ({pc}/{ms})", "color": "#fbbf24",
                "bias": "DEVELOPING", "bias_color": "#fbbf24",
                "explanation": "2 bear signals. Need 3+. Confirm volume on break.",
                "trade_idea": idea(best_p, "PUT"), "gate_pass": gate_pass}
    else:
        return {"verdict": f"FLAT — NO SIGNAL ({max(cc,pc)}/{ms})", "color": "#475569",
                "bias": "FLAT", "bias_color": "#475569",
                "explanation": "Signals mixed or absent. No trade = valid position.",
                "trade_idea": "No signal IS a signal. Sit on hands.", "gate_pass": gate_pass}

# ─── HTML HELPERS ──────────────────────────────────────────────────────────────
def h(color, txt, size=None, bold=False, extra=""):
    s_size = f"font-size:{size};" if size else ""
    s_bold = "font-weight:700;" if bold else ""
    return f'<span style="color:{color};{s_size}{s_bold}{extra}">{txt}</span>'

def sig_pill(sig, label, pf=""):
    sc = "#4ade80" if sig.get("bull") else ("#f87171" if sig.get("bear") else "#475569")
    r,g,b = int(sc[1:3],16),int(sc[3:5],16),int(sc[5:7],16)
    icon = "▲" if sig.get("bull") else ("▼" if sig.get("bear") else "–")
    return f'''<div style="background:rgba({r},{g},{b},.05);border:1px solid rgba({r},{g},{b},.2);padding:4px 6px;margin-bottom:3px">
      <div style="display:flex;justify-content:space-between;margin-bottom:1px">
        <span style="font-size:.63rem;color:{sc};font-weight:600">{label}</span>
        <span style="font-size:.55rem;color:var(--tx2)">{pf}</span>
      </div>
      <div style="font-size:.63rem;font-weight:600;color:{sc}">{icon} {sig.get("label","—")}</div>
      <div style="font-size:.58rem;color:var(--tx2);margin-top:1px">{sig.get("detail","")}</div>
    </div>'''

def chain_table(rows, is_call, price):
    if not rows: return '<tr><td colspan="6" style="color:var(--tx2)">–</td></tr>'
    out = ""
    for r in rows[:6]:
        atm = abs(r["strike"] - price) < 0.6
        kc  = "var(--bull)" if is_call else "var(--bear)"
        itm_s = "font-weight:600;" if r.get("itm") else ""
        atm_s = "background:rgba(251,191,36,.04);" if atm else ""
        vol_c = "var(--bull)" if r["vol"] > 1000 else ("var(--warn)" if r["vol"] > 200 else "var(--tx2)")
        oi_c  = "var(--bull)" if r["oi"] > 5000  else ("var(--warn)" if r["oi"] > 1000 else "var(--tx2)")
        spr_c = "var(--bull)" if r["spread"] < 0.08 else ("var(--warn)" if r["spread"] < 0.20 else "var(--bear)")
        atm_badge = ' <span style="color:var(--warn)">*</span>' if atm else ""
        out += f'''<tr style="{atm_s}">
          <td style="color:{kc};{itm_s}">${r["strike"]:.0f}{atm_badge}</td>
          <td style="color:var(--white);font-weight:600">${r["mid"]:.2f}</td>
          <td style="color:{spr_c}">${r["spread"]:.2f}</td>
          <td style="color:{vol_c}">{r["vol"]:,}</td>
          <td style="color:{oi_c}">{r["oi"]:,}</td>
          <td style="color:var(--blue)">δ{r["delta"]:.2f}</td>
        </tr>'''
    return out

def news_html(articles):
    if not articles: return '<div class="dim">No articles.</div>'
    out = ""
    for a in articles:
        src_c = "var(--bear)" if a["src"] in ("FED","VIX") else "var(--warn)" if a["src"] == "ECON" else "var(--blue)"
        out += f'''<div style="padding:4px 0;border-bottom:1px solid var(--bd2)">
          <div style="font-size:.6rem;line-height:1.4;color:var(--tx)">{a["title"]}</div>
          <div class="dim" style="margin-top:1px"><span style="color:{src_c}">{a["src"]}</span> · {a["time"]}</div>
        </div>'''
    return out

def mtf_row_html(r):
    if not r: return ""
    sc = r["bias_color"]
    ok = lambda b: f'<span style="color:{"var(--bull)" if b else "var(--bear)"}">{"▲" if b else "▼"}</span>'
    score_c = "var(--bull)" if r["score"]>=3 else ("var(--warn)" if r["score"]==2 else "var(--bear)")
    return f'''<tr>
      <td style="color:var(--white);font-weight:500">{r["label"]}</td>
      <td style="color:{sc};font-weight:500">{r["bias"]}</td>
      <td>{ok(r["bK"])}</td>
      <td>{ok(r["bR"])}</td>
      <td>{ok(r["bM"])}</td>
      <td>{ok(r["bVwap"])}</td>
      <td>{ok(r["bVol"])}</td>
      <td style="color:{score_c};font-weight:500">{r["score"]}/4</td>
    </tr>'''

# ─── RENDER ────────────────────────────────────────────────────────────────────
def render(d, opts, news_items, al, verdict, session, ct_now):
    price   = d["price"]
    chg     = d["change"]; chgp = d["change_pct"]
    cc_     = "#4ade80" if chg >= 0 else "#f87171"  # bull/bear
    arrow   = "▲" if chg >= 0 else "▼"
    ds      = ct_now.strftime("%b %d")
    ts      = ct_now.strftime("%I:%M %p CT")
    v       = verdict
    vc      = "#4ade80" if d["vol_ratio"] > 1.2 else ("#fbbf24" if d["vol_ratio"] > 0.8 else "#f87171")
    yr      = min(100, (price - d["week52l"]) / (d["week52h"] - d["week52l"]) * 100) if d["week52h"] != d["week52l"] else 50
    adx_c   = "#4ade80" if d["adx"] >= 25 else ("#fbbf24" if d["adx"] >= 20 else "#f87171")
    rvol_c  = "#4ade80" if d["rvol"] >= 1.2 else ("#fbbf24" if d["rvol"] >= 0.8 else "#f87171")
    er_val  = d["sig_kama"].get("er", 0.5)
    er_c    = "#4ade80" if er_val > 0.6 else ("#fbbf24" if er_val > 0.35 else "#f87171")

    # Nearest level
    above_lvl = next((l for l in LEVELS if l[0] > price), None)
    below_lvl = next((l for l in reversed(LEVELS) if l[0] <= price), None)

    # Session bar
    tl = ""
    for s in SESSIONS:
        t = ct_now.hour * 60 + ct_now.minute
        act = s["start"][0]*60+s["start"][1] <= t < s["end"][0]*60+s["end"][1]
        active_cls = " active" if act else ""
        bg  = f"background:{s['color']}10;border-color:{s['color']}55;" if act else ""
        nc  = s["color"] if act else "var(--tx3)"
        now = f'<div style="font-size:.52rem;color:var(--blue)">NOW</div>' if act else ""
        tl += f'<div class="sess-cell{active_cls}" style="{bg}"><div style="font-size:.52rem;color:var(--tx2)">{s["start"][0]:02d}:{s["start"][1]:02d}</div><div style="font-size:.58rem;font-weight:600;color:{nc}">{s["name"]}</div>{now}</div>'

    # MTF table rows
    mtf_html = ""
    for label in ["1H", "30m", "15m", "5m", "1m"]:
        if label in d["mtf"]:
            mtf_html += mtf_row_html(d["mtf"][label])

    # Gates
    gates = al["gates"]
    def gate_chip(name, passed):
        c = "#4ade80" if passed else "#f87171"
        bg = "rgba(74,222,128,.05)" if passed else "rgba(248,113,113,.04)"
        icon = "+" if passed else "–"
        return f'<div style="background:{bg};border:1px solid {c}22;padding:2px 4px;display:flex;align-items:center;gap:2px"><span style="color:{c};font-size:.6rem;font-weight:600">{icon}</span><span style="font-size:.55rem;color:var(--tx)">{name}</span></div>'

    gates_html = "".join(gate_chip(k, v) for k, v in gates.items())
    gate_pass = sum(1 for v in gates.values() if v)

    # Signal reasons
    def reasons_list(reasons):
        out = ""
        for r in reasons:
            c = "var(--bull)" if r.startswith("+") else "var(--tx2)"
            out += f'<div style="font-size:.58rem;color:{c};line-height:1.45;padding:1px 0">{r}</div>'
        return out

    # Candle rows
    candle_rows = ""
    for cn in d["candles"]:
        cc2 = "var(--bull)" if cn["bull"] else "var(--bear)"
        vv = cn["vol"] / 1000
        candle_rows += f'''<tr>
          <td style="color:var(--tx2)">{cn["time"]}</td>
          <td style="color:{cc2};font-weight:600">${cn["close"]:.2f}</td>
          <td style="color:{cc2}">{"▲" if cn["bull"] else "▼"}</td>
          <td style="color:var(--tx2)">{cn["high"]:.2f}</td>
          <td style="color:var(--tx2)">{cn["low"]:.2f}</td>
          <td style="color:var(--tx)">{vv:.0f}K</td>
        </tr>'''

    # Level rows
    level_rows = ""
    for (spy_p, tf, lbl, star) in LEVELS:
        dist = price - spy_p
        near = abs(dist) < 2.0
        tc   = TF_COLOR.get(tf, "var(--tx2)")
        star_s = "· " if star else ""
        bg_s = f"background:{tc}08;" if near else ""
        bl_s = f"border-left:2px solid {tc};" if near else "border-left:2px solid var(--bd);"
        dc   = "var(--bull)" if dist > 0 else "var(--bear)"
        level_rows += f'''<tr style="{bg_s}{bl_s}">
          <td style="color:{'var(--white)' if near else 'var(--tx2)'}">{star_s}{lbl}</td>
          <td style="color:{'var(--white)' if near else 'var(--tx2)'}">{tf}</td>
          <td style="color:{tc};font-weight:{'600' if near else '400'}">${spy_p:.2f}</td>
          <td style="color:{dc}">{dist:+.2f}</td>
        </tr>'''

    # On-level alert
    on_level = min(LEVELS, key=lambda x: abs(x[0] - price))
    alert_html = ""
    if abs(on_level[0] - price) <= 2.0:
        alert_html = f'<div style="padding:5px 10px;border:1px solid rgba(251,191,36,.35);background:rgba(251,191,36,.04);font-size:.63rem;color:var(--warn);text-align:center;border-radius:3px">ON LEVEL — ${on_level[0]:.2f} · {on_level[2]} ({on_level[1]}) · decision zone</div>'

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="33">
<title>Analytics Monitor</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{{
  --bg:#0a0d12;--s1:#0f1319;--s2:#141b25;--bd:#1e2738;--bd2:#171f2e;
  --tx:#94a3b8;--tx2:#64748b;--tx3:#475569;--white:#cbd5e1;
  --bull:#4ade80;--bull2:#166534;--bear:#f87171;--bear2:#7f1d1d;
  --warn:#fbbf24;--warn2:#78350f;--blue:#60a5fa;--blue2:#1e3a5f;
  --mute:#4b5563;--cyan:#22d3ee;--purple:#a78bfa;
}}
*{{margin:0;padding:0;box-sizing:border-box}}
html{{font-size:13px}}
body{{background:var(--bg);color:var(--tx);font-family:'IBM Plex Mono',monospace;line-height:1.4}}
header{{background:var(--s1);border-bottom:1px solid var(--bd);padding:6px 12px;
  display:flex;align-items:center;justify-content:space-between;gap:12px;
  position:sticky;top:0;z-index:100}}
.h-price{{font-size:1.1rem;font-weight:600;letter-spacing:.02em}}
.h-chg{{font-size:.68rem;font-weight:500}}
.h-meta{{font-size:.6rem;color:var(--tx2)}}
.main{{max-width:1600px;margin:0 auto;padding:6px 10px;display:flex;flex-direction:column;gap:5px}}
.g4{{display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:5px;align-items:start}}
.g3{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:5px;align-items:start}}
.card{{background:var(--s1);border:1px solid var(--bd);padding:8px 10px;border-radius:3px}}
.lbl{{font-size:.6rem;color:var(--tx2);letter-spacing:.08em;text-transform:uppercase;margin-bottom:4px}}
.val{{font-size:.75rem;color:var(--white);font-weight:600}}
.dim{{font-size:.6rem;color:var(--tx2)}}
.note{{font-size:.6rem;color:var(--tx)}}
.ct{{font-size:.6rem;letter-spacing:1.2px;text-transform:uppercase;color:var(--tx2);margin-bottom:5px;padding-bottom:4px;
  border-bottom:1px solid var(--bd2)}}
.sess-grid{{display:grid;grid-template-columns:repeat(5,1fr);gap:4px}}
.sess-cell{{padding:4px 6px;border:1px solid var(--bd);border-radius:2px;text-align:center}}
.sess-cell.active{{border-color:var(--blue)}}
table{{width:100%;border-collapse:collapse;table-layout:fixed}}
th{{font-size:.55rem;color:var(--tx2);font-weight:500;padding:3px 4px;
  text-align:left;border-bottom:1px solid var(--bd)}}
td{{font-size:.6rem;padding:3px 4px;border-bottom:1px solid var(--bd2);
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
tr:last-child td{{border-bottom:none}}
tr:hover td{{background:rgba(255,255,255,.012)}}
.num{{text-align:right;font-variant-numeric:tabular-nums}}
.sig-row{{display:flex;align-items:baseline;justify-content:space-between;
  padding:3px 0;border-bottom:1px solid var(--bd2);gap:6px}}
.sig-name{{font-size:.6rem;color:var(--tx2);min-width:55px}}
.sig-lbl{{font-size:.65rem;font-weight:600}}
.sig-det{{font-size:.58rem;color:var(--tx2);text-align:right}}
.gates{{display:grid;grid-template-columns:repeat(4,1fr);gap:3px;margin-top:5px}}
.bar-wrap{{height:2px;background:rgba(255,255,255,.06);border-radius:2px;overflow:hidden}}
.bar-fill{{height:100%;border-radius:2px}}
@media(max-width:1100px){{.g4{{grid-template-columns:1fr 1fr}}}}
@media(max-width:600px){{.g4{{grid-template-columns:1fr}}.g3{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<header>
  <div style="display:flex;align-items:center;gap:14px">
    <div style="font-size:.6rem;color:var(--tx2);letter-spacing:.1em">SPY · OCC</div>
    <div>
      <span class="h-price" style="color:{cc_}">${price:.2f}</span>
      <span class="h-chg" style="color:{cc_};margin-left:8px">{arrow} {abs(chg):.2f} ({abs(chgp):.2f}%)</span>
    </div>
    <div style="padding:2px 8px;border:1px solid {v['bias_color']}44;color:{v['bias_color']};font-size:.6rem;font-weight:600;letter-spacing:.06em">{v['bias']}</div>
    <div class="h-meta">ATR ${d['atr']:.2f} · RVOL {d['rvol']}x · 52W ${d['week52l']:.0f}–${d['week52h']:.0f}</div>
  </div>
  <div style="display:flex;align-items:center;gap:12px">
    <span class="h-meta">{session['name']}</span>
    <span class="h-meta">{ds} · {ts}</span>
  </div>
</header>

<div class="main">
{alert_html}

<!-- SESSION BAR -->
<div class="sess-grid">{tl}</div>

<!-- VERDICT + SIGNAL GATES -->
<div style="display:grid;grid-template-columns:1.7fr 1fr;gap:5px">
  <div style="padding:8px 12px;border:1px solid var(--bd);border-left:3px solid {v['color']};background:var(--s1);border-radius:3px">
    <div class="lbl" style="margin-bottom:2px">Status · {ts}</div>
    <div style="font-size:.88rem;font-weight:600;color:{v['color']};line-height:1.2;letter-spacing:.03em">{v['verdict']}</div>
    <div style="font-size:.63rem;color:var(--tx2);margin-top:3px">{v['explanation']}</div>
    <div style="margin-top:5px;padding:5px 8px;background:rgba(255,255,255,.025);border-left:2px solid {v['color']};font-size:.63rem;color:var(--tx);line-height:1.5">{v['trade_idea']}</div>
    <div style="margin-top:5px">
      <div class="lbl" style="margin-bottom:3px">Gates {gate_pass}/8</div>
      <div class="gates">{gates_html}</div>
    </div>
  </div>
  <div class="card">
    <div class="lbl">Parameters · 2 DTE</div>
    <div style="font-size:.63rem;line-height:2;color:var(--tx)">
      <span style="color:var(--bull)">EXP</span> {opts['expiry']} ({opts['dte']}d)<br>
      <span style="color:var(--warn)">STR</span> ATM ±1 · δ 0.42–0.52<br>
      <span style="color:var(--blue)">SPR</span> &lt;$0.08 · OI 1K+<br>
      <span style="color:var(--warn)">SC</span> 50% exit at +40–50%<br>
      <span style="color:var(--warn)">TGT</span> +80% full exit<br>
      <span style="color:var(--bear)">STP</span> –35% / ${"{:.2f}".format(price * 0.997 if d["sig_kama"].get("bull") else price * 1.003)}<br>
      <span style="color:var(--bear)">CUT</span> 10:30 CT · no exceptions<br>
    </div>
  </div>
</div>

<!-- ROW 2: 4-col main grid -->
<div class="g4">

  <!-- COL 1: Signals -->
  <div style="display:flex;flex-direction:column;gap:4px">
    <div class="card">
      <div class="ct">5m Metrics</div>
      {sig_pill(d['sig_kama'],    "KAMA/HLC3",  "PF 7.15 5m")}
      {sig_pill(d['sig_ichimoku'],"ICHIMOKU",    "1H confirm")}
      {sig_pill(d['sig_rsi'],     "RSI",         "Trebor_Namor")}
      {sig_pill(d['sig_macd'],    "MACD",        "MarcoValente")}
      {sig_pill(d['sig_vwap'],    "VWAP",        "session")}
    </div>
    <div class="card">
      <div class="ct">Breakdown</div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:3px;margin-bottom:4px">
        <div style="border:1px solid rgba(74,222,128,.25);padding:4px 6px;text-align:center">
          <div class="lbl" style="margin-bottom:2px">call</div>
          <div style="font-size:.9rem;color:var(--bull);font-weight:600">{al['call_count']}/{al['max_signals']}</div>
        </div>
        <div style="border:1px solid rgba(248,113,113,.25);padding:4px 6px;text-align:center">
          <div class="lbl" style="margin-bottom:2px">put</div>
          <div style="font-size:.9rem;color:var(--bear);font-weight:600">{al['put_count']}/{al['max_signals']}</div>
        </div>
      </div>
      <div style="font-size:.52rem;color:var(--tx2);letter-spacing:.8px;margin-bottom:2px">LONG</div>
      {reasons_list(al['call_reasons'])}
      <div style="font-size:.52rem;color:var(--tx2);letter-spacing:.8px;margin:3px 0 2px">SHORT</div>
      {reasons_list(al['put_reasons'])}
    </div>
  </div>

  <!-- COL 2: Levels + Regime -->
  <div style="display:flex;flex-direction:column;gap:4px">
    <div class="card">
      <div class="ct">Key Levels</div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:3px;margin-bottom:4px">
        <div style="padding:3px 5px;background:rgba(74,222,128,.03);border:1px solid rgba(74,222,128,.2)">
          <div class="lbl" style="margin-bottom:2px">support</div>
          <div style="font-size:.7rem;color:var(--bull);font-weight:600">${below_lvl[0]:.2f}</div>
          <div style="font-size:.5rem;color:var(--tx2)">{below_lvl[2] if below_lvl else "—"}</div>
        </div>
        <div style="padding:3px 5px;background:rgba(248,113,113,.03);border:1px solid rgba(248,113,113,.2)">
          <div class="lbl" style="margin-bottom:2px">resist</div>
          <div style="font-size:.7rem;color:var(--bear);font-weight:600">${above_lvl[0]:.2f}</div>
          <div style="font-size:.5rem;color:var(--tx2)">{above_lvl[2] if above_lvl else "—"}</div>
        </div>
      </div>
      <table>
        <tr><th>LABEL</th><th>TF</th><th>$</th><th>DIST</th></tr>
        {level_rows}
      </table>
    </div>
    <div class="card">
      <div class="ct">Regime</div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:2px;font-size:.57rem">
        <div class="card" style="padding:3px 5px;border-color:var(--bd)">
          <div style="color:var(--tx2);font-size:.5rem">MA200</div>
          <div style="color:{'var(--bull)' if d['above_e200'] else 'var(--bear)'};font-weight:600">${d['e200_d']:.2f} {'+' if d['above_e200'] else '–'}</div>
        </div>
        <div class="card" style="padding:3px 5px;border-color:var(--bd)">
          <div style="color:var(--tx2);font-size:.5rem">MA50</div>
          <div style="color:{'var(--bull)' if d['above_e50'] else 'var(--bear)'};font-weight:600">${d['e50_d']:.2f} {'+' if d['above_e50'] else '–'}</div>
        </div>
        <div class="card" style="padding:3px 5px;border-color:var(--bd)">
          <div style="color:var(--tx2);font-size:.5rem">ADX</div>
          <div style="color:{adx_c};font-weight:600">{d['adx']} {'Str' if d['adx']>=25 else 'Wk'}</div>
        </div>
        <div class="card" style="padding:3px 5px;border-color:var(--bd)">
          <div style="color:var(--tx2);font-size:.5rem">RVOL</div>
          <div style="color:{rvol_c};font-weight:600">{d['rvol']}x {'+' if d['rvol']>=1.0 else '–'}</div>
        </div>
        <div class="card" style="padding:3px 5px;border-color:var(--bd)">
          <div style="color:var(--tx2);font-size:.5rem">ER</div>
          <div style="color:{er_c};font-weight:600">{er_val:.3f} {'Tr' if er_val>0.6 else ('Tx' if er_val>0.35 else 'Ch')}</div>
        </div>
        <div class="card" style="padding:3px 5px;border-color:var(--bd)">
          <div style="color:var(--tx2);font-size:.5rem">ATR</div>
          <div style="color:var(--warn);font-weight:600">${d['atr']:.2f}</div>
        </div>
      </div>
    </div>
  </div>

  <!-- COL 3: Options Chain -->
  <div style="display:flex;flex-direction:column;gap:4px">
    <div class="card">
      <div class="ct">Chain · {opts['expiry']} · IV {opts['iv_atm']:.1f}%</div>
      <div style="font-size:.52rem;color:var(--bull);letter-spacing:1px;margin-bottom:2px">↑ CALLS</div>
      <table>
        <tr><th>STR</th><th>MID</th><th>SPR</th><th>VOL</th><th>OI</th><th>Δ</th></tr>
        {chain_table(opts['calls'], True, price)}
      </table>
      <div style="font-size:.52rem;color:var(--bear);letter-spacing:1px;margin:4px 0 2px">↓ PUTS</div>
      <table>
        <tr><th>STR</th><th>MID</th><th>SPR</th><th>VOL</th><th>OI</th><th>Δ</th></tr>
        {chain_table(opts['puts'], False, price)}
      </table>
      <div style="margin-top:3px;font-size:.5rem;color:var(--tx2)">ATM±1 · δ0.42–0.52 · vol&gt;500 · oi&gt;1K · spr&lt;.08</div>
    </div>
    <div class="card">
      <div class="ct">News</div>
      {news_html(news_items)}
    </div>
  </div>

  <!-- COL 4: MTF + Candles -->
  <div style="display:flex;flex-direction:column;gap:4px">
    <div class="card">
      <div class="ct">MTF</div>
      <table>
        <tr><th>TF</th><th>BIAS</th><th>K</th><th>R</th><th>M</th><th>V</th><th>VOL</th><th>◎</th></tr>
        {mtf_html}
      </table>
      <div style="margin-top:3px;display:grid;grid-template-columns:1fr 1fr 1fr;gap:2px;font-size:.54rem">
        <div style="padding:2px 3px;background:rgba(0,0,0,.15)">ER <strong style="color:{er_c}">{er_val:.2f}</strong></div>
        <div style="padding:2px 3px;background:rgba(0,0,0,.15)">ADX <strong style="color:{adx_c}">{d['adx']}</strong></div>
        <div style="padding:2px 3px;background:rgba(0,0,0,.15)">RVOL <strong style="color:{rvol_c}">{d['rvol']}x</strong></div>
      </div>
    </div>
    <div class="card">
      <div class="ct">5m Candles</div>
      <table>
        <tr><th>TIME</th><th>$</th><th></th><th>H</th><th>L</th><th>VOL</th></tr>
        {candle_rows}
      </table>
      <div style="margin-top:3px;border-top:1px solid var(--bd);padding-top:3px">
        <div style="font-size:.5rem;color:var(--tx2);margin-bottom:2px">52W ({yr:.0f}th %ile)</div>
        <div style="display:flex;justify-content:space-between;font-size:.52rem;color:var(--tx2);margin-bottom:2px"><span>${d['week52l']:.0f}</span><span style="color:var(--white)">${price:.2f}</span><span>${d['week52h']:.0f}</span></div>
        <div style="height:2px;background:rgba(255,255,255,.04);border-radius:2px"><div style="height:100%;width:{yr:.0f}%;background:var(--blue);border-radius:2px"></div></div>
      </div>
    </div>
    <div class="card">
      <div class="ct">Rules</div>
      <div style="font-size:.57rem;line-height:1.7;color:var(--tx)">
        <span style="color:var(--bull)">SC50</span> · +40–50%<br>
        <span style="color:var(--bull)">EXIT</span> · +80% / K-flip<br>
        <span style="color:var(--bear)">STP</span> · –35%<br>
        <span style="color:var(--warn)">TIME</span> · 10:30 CT<br>
        <span style="color:var(--warn)">TRIM</span> · MACD neg → –50%<br>
        <span style="color:var(--tx2)">NO mkt orders / avg-dn</span><br>
      </div>
    </div>
  </div>

</div>
</div>

<div style="text-align:right;padding:4px 10px;font-size:.48rem;color:var(--tx2);border-top:1px solid var(--bd)">
  upd {ct_now.strftime("%H:%M CT")}
</div>
</body></html>'''


if __name__ == "__main__":
    ct_now = datetime.datetime.now(CT)
    session = get_session(ct_now)
    print(f"⏰ {ct_now.strftime('%I:%M %p CT')} — {session['name']}")
    print("📡 Fetching SPY data...")
    d = fetch_all()
    print(f"   ${d['price']} | {d['change_pct']:+.2f}% | Vol {d['vol_ratio']}x | ATR ${d['atr']}")
    print(f"\n🔬 SIGNALS:")
    print(f"   KAMA:     {d['sig_kama']['label']} — {d['sig_kama']['detail']}")
    print(f"   Ichimoku: {d['sig_ichimoku']['label']} — {d['sig_ichimoku']['detail']}")
    print(f"   RSI:      {d['sig_rsi']['label']} — {d['sig_rsi']['detail']}")
    print(f"   MACD:     {d['sig_macd']['label']} — {d['sig_macd']['detail']}")
    print(f"   VWAP:     {d['sig_vwap']['label']} — {d['sig_vwap']['detail']}")
    print(f"   ADX: {d['adx']} | RVOL: {d['rvol']}x | ER: {d['sig_kama'].get('er',0):.3f}")
    print(f"\n📋 MTF: {list(d['mtf'].keys())}")
    opts = fetch_options(d["price"])
    print(f"⛓  {opts['expiry']} | {opts['dte']}d | IV {opts['iv_atm']:.1f}% | {len(opts['calls'])} calls / {len(opts['puts'])} puts")
    news = fetch_news()
    print(f"📰 {len(news)} articles")
    al = build_signals(d, session)
    print(f"🧠 Call {al['call_count']}/{al['max_signals']} | Put {al['put_count']}/{al['max_signals']} | Gates {sum(v for v in al['gates'].values())}/8")
    verdict = get_verdict(al, d, session, opts)
    print(f"   {verdict['verdict']}")
    html = render(d, opts, news, al, verdict, session, ct_now)
    Path("index.html").write_text(html, encoding="utf-8")
    print(f"OK {len(html):,} bytes → index.html")
