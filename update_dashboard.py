"""
NVDA Options Command Center v6 — PROVEN SYSTEM Edition
Compact layout: 2 visual rows, no wasted space, 4-col grid
"""

import yfinance as yf
import datetime, pytz, math
import feedparser
import numpy as np
from pathlib import Path

CT = pytz.timezone("America/Chicago")
ET = pytz.timezone("America/New_York")

BRANDO = [
    (288.00,"MO","$7T MCAP TARGET",   "Brando's mega bull target",         True),
    (225.00,"MO","MAJOR RESISTANCE",  "Monthly supply zone",               False),
    (200.00,"MO","KEY WALL $200",     "Tested 2x on monthly",              True),
    (191.00,"MO","YELLOW LINE $191",  "Monthly AND weekly",                True),
    (168.00,"MO","RESISTANCE",        "Monthly level",                     False),
    (153.37,"MO","DEMAND ZONE TOP",   "Institutional buy area",            True),
    (141.20,"MO","DEMAND ZONE BOT",   "Bottom of monthly demand zone",     True),
    (123.94,"MO","YELLOW LINE $124",  "Monthly yellow support pivot",      True),
    (98.43, "MO","SUPPORT",           "Monthly level",                     False),
    (76.46, "MO","DEEP DEMAND",       "Lower monthly demand zone",         True),
    (50.95, "MO","DEEP SUPPORT",      "Monthly floor",                     False),
    (250.00,"WK","RESISTANCE",        "Weekly purple line",                False),
    (225.30,"WK","MAJOR RESISTANCE",  "Weekly supply",                     False),
    (212.32,"WK","PRIOR ATH ZONE",    "Prior all-time high area",          True),
    (200.08,"WK","KEY WALL $200",     "Weekly — matches monthly $200",     True),
    (175.00,"WK","CRITICAL SUPPORT",  "Last bull defense below $180",      True),
    (149.84,"WK","SUPPORT",           "Weekly level",                      False),
    (141.23,"WK","SUPPORT",           "Weekly level",                      False),
    (132.86,"WK","SUPPORT",           "Weekly level",                      False),
    (123.12,"WK","SUPPORT",           "Weekly level",                      False),
    (115.42,"WK","SUPPORT",           "Weekly level",                      False),
    (107.50,"WK","SUPPORT",           "Weekly level",                      False),
    (95.27, "WK","SUPPORT",           "Weekly level",                      False),
    (88.19, "WK","DOUBLE BOTTOM",     "Entire 2024-25 rally base",         True),
    (200.60,"DY","RESISTANCE",        "Daily red line",                    False),
    (196.05,"DY","RESISTANCE",        "Daily red line",                    False),
    (190.40,"DY","RESISTANCE",        "Daily red line",                    False),
    (184.58,"DY","KEY RESISTANCE",    "200MA cluster zone",                True),
    (180.34,"DY","CRITICAL PIVOT",    "Floor AND ceiling — decision pt",   True),
    (175.20,"DY","SUPPORT",           "Daily support",                     False),
    (168.69,"DY","SUPPORT",           "Daily support",                     False),
    (163.67,"DY","SUPPORT",           "Daily support",                     False),
    (158.26,"DY","SUPPORT",           "Daily support",                     False),
    (153.28,"DY","SUPPORT",           "Merges with monthly demand",        True),
    (149.11,"DY","SUPPORT",           "Daily support",                     False),
    (144.22,"DY","SUPPORT",           "Daily support",                     False),
    (140.77,"DY","SUPPORT",           "Daily support",                     False),
    (136.59,"DY","SUPPORT",           "Daily support",                     False),
]

TF_COLOR = {"MO":"#b39ddb","WK":"#7c4dff","DY":"#4fc3f7"}
TF_LABEL = {"MO":"MO","WK":"WK","DY":"DY"}

def get_brando_context(price):
    res = sorted([(p,tf,l,n,k) for (p,tf,l,n,k) in BRANDO if p>price],  key=lambda x:x[0])
    sup = sorted([(p,tf,l,n,k) for (p,tf,l,n,k) in BRANDO if p<=price], key=lambda x:x[0],reverse=True)
    nearest  = min(BRANDO, key=lambda x:abs(x[0]-price))
    on_level = abs(nearest[0]-price)<=3.0 and nearest[4]
    key_res  = next((l for l in res if l[4]),res[0] if res else None)
    key_sup  = next((l for l in sup if l[4]),sup[0] if sup else None)
    if price>=200.00:   sl="🚀 ABOVE $200 WALL";  sc="#00ff9d"; sa="Hold calls. Target $212 then $225. Trail stop $196."; sb="Rejection $200-$212 → take profits. Watch reversal."
    elif price>=191.00: sl="🔥 ABOVE YELLOW LINE"; sc="#00ff9d"; sa="Bullish. Next wall $200. Buy pullbacks to $191 on 1H."; sb="Loses $191 → drops to $184.58. Calls risky."
    elif price>=184.58: sl="📈 $184–$191 ZONE";    sc="#ffd700"; sa="Break above $184.58 vol → call target $191."; sb="Rejection at $184.58 → pullback to $180.34."
    elif price>=180.34: sl="⚠️ $180.34 DECISION";  sc="#ff8c42"; sa="CALL: Holds $180.34, reclaims $184.58 → $190C target $191."; sb="PUT: Breaks $180.34 volume → $175P target $175."
    elif price>=175.00: sl="🔴 BELOW $180 BEAR";   sc="#ff3a5e"; sa="PUT active. Target $175 weekly. Stop above $180.34."; sb="Bounce to $180.34 fails → add puts."
    elif price>=153.37: sl="🔴 APPROACHING DEMAND";sc="#ff3a5e"; sa="Puts targeting $153-$141 monthly demand."; sb="Volume reversal at $153.37 → potential call entry."
    else:               sl="🆘 MONTHLY DEMAND";    sc="#ffd700"; sa="Monthly demand $141-$153 — institutional buyers."; sb="Weekly reversal candle before entering calls."
    return {"res":res[:8],"sup":sup[:8],"key_res":key_res,"key_sup":key_sup,"nearest":nearest,"on_level":on_level,"scen_label":sl,"scen_color":sc,"scen_a":sa,"scen_b":sb}

SESSIONS = [
    {"name":"PRE-MKT",      "start":(7,0),  "end":(8,30),  "color":"#5a7a99","emoji":"🌅","trade":False,"advice":"Bias only. No trades."},
    {"name":"FAKE-OUT",     "start":(8,30), "end":(9,15),  "color":"#ff8c42","emoji":"⚠️","trade":False,"advice":"FAKE-OUT ZONE. Wait for direction."},
    {"name":"CONTINUATION", "start":(9,15), "end":(10,30), "color":"#ffd700","emoji":"📈","trade":True, "advice":"Continuation if real. EMA pullback entry."},
    {"name":"DEAD ZONE",    "start":(10,30),"end":(13,0),  "color":"#ff3a5e","emoji":"💤","trade":False,"advice":"NO NEW ENTRIES. Theta bleeds."},
    {"name":"POWER WINDOW", "start":(13,0), "end":(14,30), "color":"#00ff9d","emoji":"🏆","trade":True, "advice":"BEST ENTRIES. Institutional window."},
    {"name":"CLOSING PUSH", "start":(14,30),"end":(15,0),  "color":"#4fc3f7","emoji":"🔔","trade":True, "advice":"Final push. Exit if down."},
    {"name":"AFTER HOURS",  "start":(15,0), "end":(23,59), "color":"#5a7a99","emoji":"🌙","trade":False,"advice":"Market closed. Plan tomorrow."},
]

def get_session(ct_now):
    t=ct_now.hour*60+ct_now.minute
    for s in SESSIONS:
        if s["start"][0]*60+s["start"][1]<=t<s["end"][0]*60+s["end"][1]: return s
    return SESSIONS[-1]

def ema(series,span): return series.ewm(span=span,adjust=False).mean()
def rsi(series,period=14):
    d=series.diff(); g=d.clip(lower=0).rolling(period).mean(); l=(-d.clip(upper=0)).rolling(period).mean()
    return (100-(100/(1+g/l.replace(0,np.nan)))).fillna(50)
def atr(hist,period=14):
    h,l,c=hist["High"],hist["Low"],hist["Close"]
    tr=np.maximum(h-l,np.maximum(abs(h-c.shift(1)),abs(l-c.shift(1))))
    return tr.rolling(period).mean()
def bollinger_width(series,period=20):
    mid=series.rolling(period).mean(); std=series.rolling(period).std()
    return ((mid+2*std)-(mid-2*std))/mid*100
def macd_histogram(series,fast=12,slow=26,signal=9):
    m=ema(series,fast)-ema(series,slow); return m-ema(m,signal)
def compression_score(hist):
    an=atr(hist).iloc[-1]; am=atr(hist,50).iloc[-1]
    bn=bollinger_width(hist["Close"]).iloc[-1]; bm=bollinger_width(hist["Close"],50).iloc[-1]
    ar=max(0,min(1,1-(an/am))) if am else 0; br=max(0,min(1,1-(bn/bm))) if bm else 0
    return round((ar*.5+br*.5)*100,1)
def higher_highs_lows(hist,lookback=10):
    H=hist["High"].iloc[-lookback:].values; L=hist["Low"].iloc[-lookback:].values
    hh=all(H[i]>=H[i-1] for i in range(1,len(H)) if i%2==0)
    hl=all(L[i]>=L[i-1] for i in range(1,len(L)) if i%2==0)
    return hh and hl

def calc_ichimoku(hist):
    if len(hist)<60: return {"bull":False,"bear":False,"label":"N/A","detail":"No data","tenkan":None,"kijun":None,"spanA":None,"spanB":None,"cloud_bull":None,"cloud_top":None,"cloud_bottom":None}
    c=hist["Close"]; h=hist["High"]; l=hist["Low"]
    tenkan=(h.rolling(9).max()+l.rolling(9).min())/2; kijun=(h.rolling(26).max()+l.rolling(26).min())/2
    spanA=(tenkan+kijun)/2; spanB=(h.rolling(52).max()+l.rolling(52).min())/2
    tn=float(tenkan.iloc[-1]); kn=float(kijun.iloc[-1]); an=float(spanA.iloc[-1]); bn=float(spanB.iloc[-1]); pn=float(c.iloc[-1])
    ct=max(an,bn); cb=min(an,bn); cbull=an>bn
    above=pn>ct; below=pn<cb; tkb=tn>kn; tkbr=tn<kn
    bull=above and tkb; bear=below and tkbr
    if bull:   lbl="✅ BULL"; det=f"Above cloud·T>{kn:.1f}"
    elif bear: lbl="✅ BEAR"; det=f"Below cloud·T<{kn:.1f}"
    elif above: lbl="⚠️ WEAK BULL"; det="Above cloud,T≤K"
    elif below: lbl="⚠️ WEAK BEAR"; det="Below cloud,T≥K"
    else:       lbl="⏸ IN CLOUD"; det=f"${cb:.1f}–${ct:.1f}"
    return {"bull":bull,"bear":bear,"label":lbl,"detail":det,"tenkan":round(tn,2),"kijun":round(kn,2),"spanA":round(an,2),"spanB":round(bn,2),"cloud_bull":cbull,"cloud_top":round(ct,2),"cloud_bottom":round(cb,2)}

def calc_kama(hist,length=10,fast_sc=2,slow_sc=30):
    if len(hist)<length+10: return {"bull":False,"bear":False,"label":"N/A","detail":"No data","kama_val":None,"er":None,"direction":None,"dir_color":"#5a7a99"}
    hlc3=(hist["High"]+hist["Low"]+hist["Close"])/3
    ch=abs(hlc3-hlc3.shift(length)); vol=abs(hlc3-hlc3.shift(1)).rolling(length).sum()
    er=(ch/vol.replace(0,np.nan)).fillna(0)
    fc=2./(fast_sc+1); sc_=2./(slow_sc+1); sc=(er*(fc-sc_)+sc_)**2
    kv=np.zeros(len(hlc3)); ha=hlc3.values; sa=sc.values; kv[0]=ha[0]
    for i in range(1,len(ha)):
        kv[i]=kv[i-1] if (np.isnan(sa[i]) or np.isnan(ha[i])) else kv[i-1]+sa[i]*(ha[i]-kv[i-1])
    kn=kv[-1]; kp=kv[-2]; hn=float(hlc3.iloc[-1]); en=float(er.iloc[-1])
    kr=kn>kp; kf=kn<kp
    bull=hn>kn and kr; bear=hn<kn and kf
    d_="RISING" if kr else ("FALLING" if kf else "FLAT"); dc="#00ff9d" if kr else ("#ff3a5e" if kf else "#ffd700")
    if bull:   lbl="✅ BULL"; det=f"HLC3>{kn:.1f}·KAMA↑·ER{en:.2f}"
    elif bear: lbl="✅ BEAR"; det=f"HLC3<{kn:.1f}·KAMA↓·ER{en:.2f}"
    elif hn>kn: lbl="⚠️ WEAK BULL"; det="Above KAMA,not rising"
    elif hn<kn: lbl="⚠️ WEAK BEAR"; det="Below KAMA,not falling"
    else:       lbl="⏸ NEUTRAL";   det="On KAMA"
    return {"bull":bull,"bear":bear,"label":lbl,"detail":det,"kama_val":round(kn,2),"er":round(en,3),"direction":d_,"dir_color":dc}

def calc_rsi_signal(hist,period=14,bull_level=45,bear_level=55,oversold=30,overbought=70):
    if len(hist)<period+5: return {"bull":False,"bear":False,"label":"N/A","detail":"No data","rsi_val":None,"rsi_prev":None,"cross_type":None}
    rs=rsi(hist["Close"],period); rn=float(rs.iloc[-1]); rp=float(rs.iloc[-2])
    ca45=rp<bull_level and rn>=bull_level; ob=rn<oversold and rn>rp
    cb55=rp>bear_level and rn<=bear_level; obr=rn>overbought and rn<rp
    bull=ca45 or ob; bear=cb55 or obr
    ct=None
    if ca45: ct="CROSS↑45"
    elif ob:  ct="OB BOUNCE"
    elif cb55: ct="CROSS↓55"
    elif obr: ct="OB REJECT"
    zone="OB" if rn>overbought else ("OS" if rn<oversold else "MID")
    if bull:   lbl="✅ BULL"; det=f"RSI{rp:.0f}→{rn:.0f} {ct}"
    elif bear: lbl="✅ BEAR"; det=f"RSI{rp:.0f}→{rn:.0f} {ct}"
    else:      lbl=f"⏸ {zone}"; det=f"RSI{rn:.0f} wait 45/55"
    return {"bull":bull,"bear":bear,"label":lbl,"detail":det,"rsi_val":round(rn,1),"rsi_prev":round(rp,1),"cross_type":ct}

def calc_macd_rsi_combo(hist,period=14):
    if len(hist)<30: return {"bull":False,"bear":False,"label":"N/A","detail":"No data","macd_h":None,"rsi_val":None}
    c=hist["Close"]; mh=macd_histogram(c); rs=rsi(c,period)
    mn=float(mh.iloc[-1]); mp=float(mh.iloc[-2]); rn=float(rs.iloc[-1])
    mcb=mp<0 and mn>=0; mcbr=mp>0 and mn<=0
    bull=mcb and 40<rn<65; bear=mcbr and 35<rn<60
    if bull:   lbl="✅ BULL"; det=f"MACD↑0·RSI{rn:.0f}"
    elif bear: lbl="✅ BEAR"; det=f"MACD↓0·RSI{rn:.0f}"
    else:      lbl="⏸ WAIT"; det=f"MACD{mn:+.3f}·RSI{rn:.0f}"
    return {"bull":bull,"bear":bear,"label":lbl,"detail":det,"macd_h":round(mn,3),"rsi_val":round(rn,1)}

def detect_rsi_divergence(hist,rsi_period=5,lookback=3):
    c=hist["Close"]
    if len(c)<rsi_period+lookback+5: return _div_none()
    rs=rsi(c,rsi_period); pv=c.values; rv=rs.values
    cp=pv[-1]; pp=pv[-(lookback+1)]; cr=rv[-1]; pr=rv[-(lookback+1)]
    pu=cp>pp; pd=cp<pp; ru=cr>pr; rd=cr<pr
    sp=abs(cr-pr)
    st="strong" if sp>=8 else ("moderate" if sp>=4 else "weak")
    if pd and ru and cr<50: return {"divergence_type":"regular_bull","label":f"📈 REG BULL ({st})","detail":f"P↓ RSI↑{pr:.0f}→{cr:.0f}","strength":st,"color":"#00ff9d","confirms_bull":True,"confirms_bear":False,"exit_warning_bull":False,"exit_warning_bear":True}
    if pu and rd and cr<55: return {"divergence_type":"hidden_bull","label":f"🔷 HID BULL ({st})","detail":f"P↑ RSI↓{pr:.0f}→{cr:.0f}","strength":st,"color":"#4fc3f7","confirms_bull":True,"confirms_bear":False,"exit_warning_bull":False,"exit_warning_bear":True}
    if pu and rd and cr>50: return {"divergence_type":"regular_bear","label":f"📉 REG BEAR ({st})","detail":f"P↑ RSI↓{pr:.0f}→{cr:.0f}","strength":st,"color":"#ff3a5e","confirms_bull":False,"confirms_bear":True,"exit_warning_bull":True,"exit_warning_bear":False}
    if pd and ru and cr>45: return {"divergence_type":"hidden_bear","label":f"🔻 HID BEAR ({st})","detail":f"P↓ RSI↑{pr:.0f}→{cr:.0f}","strength":st,"color":"#ff8c42","confirms_bull":False,"confirms_bear":True,"exit_warning_bull":True,"exit_warning_bear":False}
    return _div_none()

def _div_none():
    return {"divergence_type":None,"label":"No divergence","detail":"","strength":None,"color":"#5a7a99","confirms_bull":False,"confirms_bear":False,"exit_warning_bull":False,"exit_warning_bear":False}

def tf_ind(h,label,price):
    c=h["Close"]
    e8=round(float(ema(c,8).iloc[-1]),2); e21=round(float(ema(c,21).iloc[-1]),2); e50=round(float(ema(c,50).iloc[-1]),2)
    r=round(float(rsi(c).iloc[-1]),1); mh=macd_histogram(c)
    mn=round(float(mh.iloc[-1]),3); mp=round(float(mh.iloc[-2]),3)
    vn=int(h["Volume"].iloc[-1]); vm=float(h["Volume"].rolling(20).mean().iloc[-1]) or 1
    vr=round(vn/vm,2); hh=higher_highs_lows(h)
    return {"label":label,"e8":e8,"e21":e21,"e50":e50,"rsi":r,"macd_h":mn,"macd_h_prev":mp,"vol_ratio":vr,"hh_hl":hh,
            "above_cloud":price>max(e21,e50),"below_cloud":price<min(e21,e50),"fast_bull":e8>e21,
            "macd_bull":mn>0,"macd_turning":mn>0 and mp<0,"macd_turning_bear":mn<0 and mp>0,
            "rsi_expansion_bull":r>60,"rsi_expansion_bear":r<40,"vol_expanded":vr>=1.5}

def fetch_all():
    nvda=yf.Ticker("NVDA"); spy=yf.Ticker("SPY"); info=nvda.info
    d1=nvda.history(period="1y",interval="1d"); h4=nvda.history(period="60d",interval="4h")
    h1=nvda.history(period="30d",interval="1h"); m15=nvda.history(period="5d",interval="15m")
    m5=nvda.history(period="2d",interval="5m"); d1_3m=nvda.history(period="3mo",interval="1d")
    spy_d=spy.history(period="6mo",interval="1d")
    price=float(info.get("currentPrice") or info.get("regularMarketPrice") or d1["Close"].iloc[-1])
    prev_close=float(info.get("previousClose") or d1["Close"].iloc[-2])
    volume=int(info.get("regularMarketVolume") or d1["Volume"].iloc[-1])
    avg_vol=int(info.get("averageVolume") or d1["Volume"].mean())
    week52h=float(info.get("fiftyTwoWeekHigh") or d1["High"].max())
    week52l=float(info.get("fiftyTwoWeekLow") or d1["Low"].min())
    spy_e50=float(ema(spy_d["Close"],50).iloc[-1]); spy_price=float(spy_d["Close"].iloc[-1])
    spy_bull=spy_price>spy_e50
    comp=compression_score(d1_3m) if len(d1_3m)>=50 else 0
    atr_val=round(float(atr(d1).iloc[-1]),2); vol20ma=float(d1["Volume"].rolling(20).mean().iloc[-1])
    sig_ichimoku=calc_ichimoku(m15); sig_kama=calc_kama(h4)
    sig_rsi=calc_rsi_signal(h1); sig_macd_combo=calc_macd_rsi_combo(h1)
    h1_div=detect_rsi_divergence(h1,rsi_period=5,lookback=3) if len(h1)>=20 else _div_none()
    daily_tf=tf_ind(d1,"DAILY",price) if len(d1)>=50 else None
    h4_tf=tf_ind(h4,"4H",price)       if len(h4)>=50 else None
    h1_tf=tf_ind(h1,"1H",price)       if len(h1)>=50 else None
    rc=[]
    if len(m5)>=4:
        for idx,row in m5.tail(8).iterrows():
            try: ts=idx.tz_convert(CT).strftime("%I:%M %p")
            except: ts=str(idx).replace("T"," ")[11:16]
            co,cc=float(row["Open"]),float(row["Close"])
            rc.append({"time":ts,"open":round(co,2),"high":round(float(row["High"]),2),"low":round(float(row["Low"]),2),"close":round(cc,2),"volume":int(row["Volume"]),"bull":cc>=co})
    id_high=round(float(m5["High"].max()),2) if len(m5) else price
    id_low =round(float(m5["Low"].min()),2)  if len(m5) else price
    return {"price":price,"prev_close":round(prev_close,2),"change":round(price-prev_close,2),"change_pct":round((price-prev_close)/prev_close*100,2),
            "volume":volume,"avg_vol":avg_vol,"vol_ratio":round(volume/avg_vol,2) if avg_vol else 1.0,"vol_20ma":round(vol20ma/1e6,1),
            "week52h":week52h,"week52l":week52l,"atr":atr_val,"compression":comp,
            "sig_ichimoku":sig_ichimoku,"sig_kama":sig_kama,"sig_rsi":sig_rsi,"sig_macd_combo":sig_macd_combo,"h1_div":h1_div,
            "daily":daily_tf,"h4":h4_tf,"h1":h1_tf,"spy_bull":spy_bull,"spy_price":round(spy_price,2),"spy_e50":round(spy_e50,2),
            "recent_candles":rc,"id_high":id_high,"id_low":id_low,"d1":d1,"d1_3m":d1_3m}

def fetch_options(price):
    tk=yf.Ticker("NVDA"); exps=tk.options; today=datetime.datetime.now(ET).date()
    target=today+datetime.timedelta(days=25)
    chosen=next((e for e in exps if datetime.datetime.strptime(e,"%Y-%m-%d").date()>=target),exps[-1] if exps else None)
    if not chosen: return {"expiry":"N/A","calls":[],"puts":[],"iv_30":50,"dte":30,"exp_move":0}
    chain=tk.option_chain(chosen); dte=(datetime.datetime.strptime(chosen,"%Y-%m-%d").date()-today).days
    calls_raw,puts_raw=[],[]; iv_atm=0.50
    for _,r in chain.calls.iterrows():
        s=float(r["strike"])
        if abs(s-price)<=25: calls_raw.append({"strike":s,"price":round(float(r.get("lastPrice",0) or r.get("ask",0) or 0),2),"bid":round(float(r.get("bid",0) or 0),2),"ask":round(float(r.get("ask",0) or 0),2),"delta":round(float(r.get("delta",0) or 0),2),"theta":round(float(r.get("theta",0) or 0),3),"iv":round(float(r.get("impliedVolatility",0.5) or 0.5)*100,1),"volume":int(r.get("volume",0) or 0),"oi":int(r.get("openInterest",0) or 0)})
    for _,r in chain.puts.iterrows():
        s=float(r["strike"])
        if abs(s-price)<=20 and s<=price+5: puts_raw.append({"strike":s,"price":round(float(r.get("lastPrice",0) or r.get("ask",0) or 0),2),"bid":round(float(r.get("bid",0) or 0),2),"ask":round(float(r.get("ask",0) or 0),2),"delta":round(float(r.get("delta",0) or 0),2),"theta":round(float(r.get("theta",0) or 0),3),"iv":round(float(r.get("impliedVolatility",0.5) or 0.5)*100,1),"volume":int(r.get("volume",0) or 0),"oi":int(r.get("openInterest",0) or 0)})
    atm_c=[c for c in calls_raw if abs(c["strike"]-price)<5]
    if atm_c: iv_atm=atm_c[0]["iv"]/100
    calls_raw.sort(key=lambda x:x["strike"]); puts_raw.sort(key=lambda x:x["strike"],reverse=True)
    return {"expiry":chosen,"calls":calls_raw[:7],"puts":puts_raw[:4],"iv_30":round(iv_atm*100,1),"dte":dte,"exp_move":round(price*iv_atm*math.sqrt(dte/365),2)}

def fetch_news():
    urls=["https://feeds.finance.yahoo.com/rss/2.0/headline?s=NVDA&region=US&lang=en-US","https://finance.yahoo.com/rss/headline?s=NVDA"]
    bull=["beat","surge","rally","upgrade","buy","growth","record","strong","profit","bullish","target","raise","deal","expand","partnership"]
    bear=["miss","drop","fall","downgrade","sell","cut","loss","bearish","concern","risk","decline","weak","ban","tariff","probe","lawsuit"]
    out=[]
    for url in urls:
        try:
            feed=feedparser.parse(url)
            for e in feed.entries[:8]:
                t=e.get("title",""); txt=(t+" "+e.get("summary","")).lower()
                bs=sum(1 for w in bull if w in txt); bs2=sum(1 for w in bear if w in txt)
                out.append({"title":t[:100],"published":e.get("published","")[:22],"link":e.get("link","#"),"sentiment":"bull" if bs>bs2 else ("bear" if bs2>bs else "neut")})
            if out: break
        except: continue
    return out[:6]

USE_4TH_SIGNAL=True

def build_signals(d,opts,session,bctx):
    price=d["price"]; s1=d["sig_ichimoku"]; s2=d["sig_kama"]; s3=d["sig_rsi"]; s4=d["sig_macd_combo"]
    div=d["h1_div"]; ms=4 if USE_4TH_SIGNAL else 3
    cc=0; cr=[]
    if s1["bull"]: cc+=1; cr.append(f'✅ S1 ICHI: {s1["detail"]}')
    else:          cr.append(f'❌ S1 ICHI: {s1["detail"]}')
    if s2["bull"]: cc+=1; cr.append(f'✅ S2 KAMA: {s2["detail"]}')
    else:          cr.append(f'❌ S2 KAMA: {s2["detail"]}')
    if s3["bull"]: cc+=1; cr.append(f'✅ S3 RSI: {s3["detail"]}')
    else:          cr.append(f'❌ S3 RSI: {s3["detail"]}')
    if USE_4TH_SIGNAL:
        if s4["bull"]: cc+=1; cr.append(f'✅ S4 MACD: {s4["detail"]}')
        else:          cr.append(f'❌ S4 MACD: {s4["detail"]}')
    cx=[]
    if price>=180.34: cx.append("✅ Above $180.34 bias")
    else:             cx.append("⚠️ Below $180.34 — vs Brando")
    if d["spy_bull"]: cx.append("✅ SPY tailwind")
    else:             cx.append("⚠️ SPY headwind")
    pu=d["change"]>=0
    if pu and d["vol_ratio"]>=1.5:    cx.append(f"✅ Vol {d['vol_ratio']}x up day")
    elif not pu and d["vol_ratio"]<1.0: cx.append(f"✅ Low-vol pullback {d['vol_ratio']}x")
    elif not pu and d["vol_ratio"]>=1.5: cx.append(f"❌ Vol {d['vol_ratio']}x selling")
    else: cx.append(f"⚠️ Vol {d['vol_ratio']}x neutral")
    cx.append(f"{'✅' if session['trade'] else '⚠️'} {session['name']}")
    if div["confirms_bull"]:       cx.append(f"✅ DIV {div['label']}")
    elif div["exit_warning_bull"]: cx.append(f"🚨 DIV EXIT {div['label']}")
    pc=0; pr=[]
    if s1["bear"]: pc+=1; pr.append(f'✅ S1 ICHI: {s1["detail"]}')
    else:          pr.append(f'❌ S1 ICHI: {s1["detail"]}')
    if s2["bear"]: pc+=1; pr.append(f'✅ S2 KAMA: {s2["detail"]}')
    else:          pr.append(f'❌ S2 KAMA: {s2["detail"]}')
    if s3["bear"]: pc+=1; pr.append(f'✅ S3 RSI: {s3["detail"]}')
    else:          pr.append(f'❌ S3 RSI: {s3["detail"]}')
    if USE_4TH_SIGNAL:
        if s4["bear"]: pc+=1; pr.append(f'✅ S4 MACD: {s4["detail"]}')
        else:          pr.append(f'❌ S4 MACD: {s4["detail"]}')
    px=[]
    if price<180.34: px.append("✅ Below $180.34 bias")
    else:            px.append("⚠️ Above $180.34 — vs Brando")
    if not d["spy_bull"]: px.append("✅ SPY tailwind")
    else:                 px.append("⚠️ SPY headwind")
    pd_=d["change"]<0
    if pd_ and d["vol_ratio"]>=1.5:   px.append(f"✅ Vol {d['vol_ratio']}x selling")
    elif pd_ and d["vol_ratio"]<1.0:  px.append(f"❌ Low-vol decline — dead cat")
    elif not pd_ and d["vol_ratio"]>=1.5: px.append(f"❌ Vol {d['vol_ratio']}x up day")
    else: px.append(f"⚠️ Vol {d['vol_ratio']}x neutral")
    px.append(f"{'✅' if session['trade'] else '⚠️'} {session['name']}")
    if div["confirms_bear"]:       px.append(f"✅ DIV {div['label']}")
    elif div["exit_warning_bear"]: px.append(f"🚨 DIV EXIT {div['label']}")
    ct=next((c for c in opts["calls"] if 3.50<=c["price"]<=6.50),None)
    pt=next((p for p in opts["puts"]  if 3.50<=p["price"]<=6.50),None)
    def rrc(opt,sig_count):
        if not opt: return None
        e=opt["price"]; st=round(e*0.65,2); tg=round(e*1.60,2)
        ri=round(e-st,2); rw=round(tg-e,2); rr=round(rw/ri,1) if ri else 0
        b=1000 if sig_count>=4 else (750 if sig_count>=3 else 500)
        cn=max(1,int(b/(e*100))); ml=round(cn*e*100*0.35,0)
        return {"entry":e,"stop":st,"target":tg,"risk":ri,"reward":rw,"rr":rr,"contracts":cn,"max_risk":int(ml),"budget":b}
    return {"call_count":cc,"put_count":pc,"max_signals":ms,"call_reasons":cr,"put_reasons":pr,"call_context":cx,"put_context":px,"call_t":ct,"put_t":pt,"call_rr":rrc(ct,cc),"put_rr":rrc(pt,pc),"h1_div":div}

def get_verdict(al,d,session,opts,bctx):
    cs=al["call_count"]; ps=al["put_count"]; ms=al["max_signals"]; div=al["h1_div"]
    def sl(c): return "$1,000" if c>=4 else ("$750" if c>=3 else "$500")
    if not session["trade"]: return {"verdict":f"💤 {session['name']} — NO NEW TRADES","color":"#ff3a5e","bias":"WAIT","bias_color":"#ff8c42","explanation":session["advice"],"trade_idea":"Next: Power Window 1:00 PM CT."}
    if cs>=3:
        ct=al["call_t"]; rr=al["call_rr"]; cv="MAX" if cs>=4 else "HIGH"
        dn=f" · {div['label']}" if div["confirms_bull"] else ""
        return {"verdict":f"✅ CALL — {cv} ({cs}/{ms})","color":"#00ff9d","bias":"STRONG BULL","bias_color":"#00ff9d","explanation":f"{cs} signals. Size {sl(cs)}. Target ${bctx['key_res'][0] if bctx['key_res'] else '—'}.{dn}","trade_idea":f"${ct['strike']:.0f}C {opts['expiry']} ~${ct['price']:.2f} | Stop ${rr['stop']} | Tgt ${rr['target']} | R:R {rr['rr']}:1 · {rr['contracts']}ct" if ct and rr else "Check chain."}
    elif cs==2:
        ct=al["call_t"]; rr=al["call_rr"]
        return {"verdict":f"🟢 CALL — GOOD ({cs}/{ms})","color":"#00c87a","bias":"BULLISH","bias_color":"#00ff9d","explanation":f"2/{ms} signals. $500 size. Confirm context.","trade_idea":f"${ct['strike']:.0f}C {opts['expiry']} ~${ct['price']:.2f} | Stop ${rr['stop']} | Tgt ${rr['target']}" if ct and rr else "Check chain."}
    elif ps>=3:
        pt=al["put_t"]; rr=al["put_rr"]; cv="MAX" if ps>=4 else "HIGH"
        dn=f" · {div['label']}" if div["confirms_bear"] else ""
        return {"verdict":f"🔴 PUT — {cv} ({ps}/{ms})","color":"#ff3a5e","bias":"STRONG BEAR","bias_color":"#ff3a5e","explanation":f"{ps} signals bearish. Size {sl(ps)}. Target ${bctx['key_sup'][0] if bctx['key_sup'] else '—'}.{dn}","trade_idea":f"${pt['strike']:.0f}P {opts['expiry']} ~${pt['price']:.2f} | Stop ${rr['stop']} | Tgt ${rr['target']} | R:R {rr['rr']}:1 · {rr['contracts']}ct" if pt and rr else "Check chain."}
    elif ps==2:
        pt=al["put_t"]; rr=al["put_rr"]
        return {"verdict":f"🟠 PUT — GOOD ({ps}/{ms})","color":"#ff8c42","bias":"BEARISH","bias_color":"#ff3a5e","explanation":f"2/{ms} signals. Volume must confirm (no dead cat).","trade_idea":f"${pt['strike']:.0f}P {opts['expiry']} ~${pt['price']:.2f} | Stop ${rr['stop']} | Tgt ${rr['target']}" if pt and rr else "Watch breakdown."}
    elif cs==1 and ps==1: return {"verdict":"⏸ MIXED — NO TRADE","color":"#ff8c42","bias":"NEUTRAL","bias_color":"#ffd700","explanation":f"Conflicting. {bctx['scen_label']}","trade_idea":f"A: {bctx['scen_a']} | B: {bctx['scen_b']}"}
    elif cs==1: return {"verdict":f"⏸ CALL WATCH ({cs}/{ms})","color":"#ffd700","bias":"DEVELOPING","bias_color":"#ffd700","explanation":"1 signal. Need 2+.","trade_idea":f"A: {bctx['scen_a']} | B: {bctx['scen_b']}"}
    elif ps==1: return {"verdict":f"⏸ PUT WATCH ({ps}/{ms})","color":"#ffd700","bias":"DEVELOPING","bias_color":"#ffd700","explanation":"1 bear signal. Need 2+.","trade_idea":f"A: {bctx['scen_a']} | B: {bctx['scen_b']}"}
    else: return {"verdict":"⏸ FLAT — NO SIGNAL","color":"#5a7a99","bias":"FLAT","bias_color":"#5a7a99","explanation":f"0 signals. {bctx['scen_label']}.","trade_idea":"No signal IS a signal."}


# ─────────────────────────────────────────────
# RENDER — Compact 2-row layout
# Row 1: Header → Session → Verdict
# Row 2: 4-col grid  [Signals+Div | Brando+Scenario | Call+Put | Chain+News]
# Row 3: 4-col grid  [MTF | Regime | Candles | Entry Model]
# ─────────────────────────────────────────────
def render(d,opts,news,al,verdict,session,bctx,ct_now):
    price=d["price"]; chg=d["change"]; chgp=d["change_pct"]
    cc="#00ff9d" if chg>=0 else "#ff3a5e"; arrow="▲" if chg>=0 else "▼"
    vm=d["volume"]/1e6; am=d["avg_vol"]/1e6
    vc="#00ff9d" if d["vol_ratio"]>1.2 else ("#ffd700" if d["vol_ratio"]>0.8 else "#ff3a5e")
    yr=min(100,(price-d["week52l"])/(d["week52h"]-d["week52l"])*100) if d["week52h"]!=d["week52l"] else 50
    v=verdict; sc="#00ff9d" if d["spy_bull"] else "#ff3a5e"; sl="BULL✓" if d["spy_bull"] else "BEAR✗"
    ds=ct_now.strftime("%b %d"); ts=ct_now.strftime("%I:%M %p CT")
    comp=d["compression"]; cc2="#00ff9d" if comp>70 else ("#ffd700" if comp>40 else "#5a7a99")
    div=al["h1_div"]; ms=al["max_signals"]
    s1=d["sig_ichimoku"]; s2=d["sig_kama"]; s3=d["sig_rsi"]; s4=d["sig_macd_combo"]

    # Session bar
    tl=""
    for s in SESSIONS[:-1]:
        t=ct_now.hour*60+ct_now.minute; act=s["start"][0]*60+s["start"][1]<=t<s["end"][0]*60+s["end"][1]
        bg=f"background:{s['color']}20;border:1px solid {s['color']}99;" if act else "background:rgba(255,255,255,.018);border:1px solid var(--bd);"
        nc=s["color"] if act else "var(--muted)"
        ndot=f'<div style="font-size:.5rem;color:{s["color"]}">●NOW</div>' if act else ""
        tl+=f'<div style="{bg}padding:3px 5px;text-align:center"><div style="font-size:.54rem;color:var(--muted)">{s["start"][0]:02d}:{s["start"][1]:02d}</div><div style="font-size:.6rem;font-weight:700;color:{nc};white-space:nowrap">{s["emoji"]} {s["name"]}</div>{ndot}</div>'

    # Compact signal card
    def sig_card(num,name,sig,pf,tf_,accent):
        sc2="#00ff9d" if sig.get("bull") else ("#ff3a5e" if sig.get("bear") else "#5a7a99")
        r,g,b=int(sc2[1:3],16),int(sc2[3:5],16),int(sc2[5:7],16)
        return f'''<div style="background:rgba({r},{g},{b},.06);border:1px solid rgba({r},{g},{b},.35);padding:6px 7px">
          <div style="display:flex;justify-content:space-between;margin-bottom:2px">
            <span style="font-size:.56rem;color:{accent};font-weight:700">{num} {name}</span>
            <span style="font-size:.52rem;color:var(--muted);background:rgba(255,255,255,.05);padding:1px 4px">{tf_} {pf}</span>
          </div>
          <div style="font-size:.73rem;font-weight:700;color:{sc2}">{sig["label"]}</div>
          <div style="font-size:.6rem;color:var(--text);margin-top:1px;line-height:1.35;opacity:.9">{sig["detail"]}</div>
        </div>'''

    # Brando mini table (4 res + price row + 4 sup)
    def brando_mini():
        r4=bctx["res"][:4]; s4_=bctx["sup"][:4]; rows=""
        for (p,tf,lbl,note,key) in r4:
            dist=round(p-price,2); tc=TF_COLOR.get(tf,"#aaa"); bg="background:rgba(255,215,0,.04);" if key else ""
            lc="#ffd700" if key else "#ff6b6b"; st="★" if key else " "
            rows+=f'<tr style="{bg}"><td style="color:{tc};font-size:.56rem">{TF_LABEL[tf]}</td><td style="color:{lc};font-weight:700;font-size:.65rem">${p:.0f}</td><td style="color:var(--muted);font-size:.56rem;overflow:hidden;text-overflow:ellipsis">{st}{lbl[:16]}</td><td style="color:var(--muted);font-size:.56rem;text-align:right">+{dist:.1f}</td></tr>'
        ons=f'⚡{bctx["nearest"][2][:10]}' if bctx["on_level"] else f'H{d["id_high"]}L{d["id_low"]}'
        rows+=f'<tr style="background:rgba(255,215,0,.09)"><td colspan="2" style="color:#ffd700;font-weight:700;font-size:.7rem">▶${price:.2f}</td><td colspan="2" style="color:#ffd700;font-size:.55rem;text-align:right">{ons}</td></tr>'
        for (p,tf,lbl,note,key) in s4_:
            dist=round(price-p,2); tc=TF_COLOR.get(tf,"#aaa"); bg="background:rgba(255,215,0,.04);" if key else ""
            lc="#ffd700" if key else "#00c87a"; st="★" if key else " "
            rows+=f'<tr style="{bg}"><td style="color:{tc};font-size:.56rem">{TF_LABEL[tf]}</td><td style="color:{lc};font-weight:700;font-size:.65rem">${p:.0f}</td><td style="color:var(--muted);font-size:.56rem;overflow:hidden;text-overflow:ellipsis">{st}{lbl[:16]}</td><td style="color:var(--muted);font-size:.56rem;text-align:right">-{dist:.1f}</td></tr>'
        return rows

    # Signal column (reasons + context + R:R)
    def sig_col(reasons,context,count,ms_,direction):
        gc="#00ff9d" if direction=="call" else "#ff3a5e"; pct=round(count/ms_*100)
        h=f'<div style="margin-bottom:5px"><div style="display:flex;justify-content:space-between;font-size:.58rem;color:var(--muted);margin-bottom:2px"><span>SIGNALS</span><span style="color:{gc};font-weight:700">{count}/{ms_}</span></div><div style="height:3px;background:rgba(255,255,255,.05);border-radius:2px"><div style="height:100%;width:{pct}%;background:{gc};border-radius:2px"></div></div></div>'
        for r in reasons:
            rc2="#00ff9d" if r.startswith("✅") else ("#ff3a5e" if r.startswith("❌") else "#ffd700")
            h+=f'<div style="font-size:.62rem;color:{rc2};padding:2px 0;border-bottom:1px solid rgba(255,255,255,.04);line-height:1.3">{r}</div>'
        h+='<div style="font-size:.54rem;color:var(--muted);letter-spacing:.5px;margin:5px 0 2px">CONTEXT</div>'
        for cx in context:
            rc2="#00ff9d" if cx.startswith("✅") else ("#ff3a5e" if cx.startswith(("❌","🚨")) else "#ffd700")
            h+=f'<div style="font-size:.6rem;color:{rc2};padding:1px 0;line-height:1.3">{cx}</div>'
        return h

    def rr_mini(rr,direction):
        if not rr: return '<div style="font-size:.63rem;color:var(--muted);padding:6px;text-align:center">No ATM $3–6 option</div>'
        bc="#00ff9d" if direction=="call" else "#ff3a5e"; rc2="#00ff9d" if rr["rr"]>=2 else ("#ffd700" if rr["rr"]>=1.5 else "#ff3a5e")
        return f'''<div style="background:rgba(255,255,255,.02);border:1px solid {bc}44;padding:6px;margin-top:5px">
          <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:3px;text-align:center;margin-bottom:4px">
            <div><div style="font-size:.52rem;color:var(--muted)">ENTRY</div><div style="font-size:.76rem;color:var(--white);font-weight:700">${rr["entry"]:.2f}</div></div>
            <div><div style="font-size:.52rem;color:var(--muted)">STOP</div><div style="font-size:.76rem;color:var(--red);font-weight:700">${rr["stop"]:.2f}</div></div>
            <div><div style="font-size:.52rem;color:var(--muted)">TARGET</div><div style="font-size:.76rem;color:var(--green);font-weight:700">${rr["target"]:.2f}</div></div>
            <div><div style="font-size:.52rem;color:var(--muted)">R:R</div><div style="font-size:.76rem;font-weight:700;color:{rc2}">{rr["rr"]}:1</div></div>
          </div>
          <div style="display:flex;justify-content:space-between;font-size:.56rem;color:var(--muted)">
            <span>${rr["budget"]}·{rr["contracts"]}ct</span><span style="color:var(--red)">max ${rr["max_risk"]}</span>
          </div></div>'''

    def chain_rows(items):
        rows=""
        for c in items:
            atm=' style="background:rgba(255,215,0,.05)"' if abs(c["strike"]-price)<4 else ""
            tgt="🎯" if 3.50<=c["price"]<=6.50 else ""
            sp=round(c["ask"]-c["bid"],2); sc2="#00ff9d" if sp<0.20 else ("#ffd700" if sp<0.40 else "#ff3a5e")
            liq="⚠" if c["volume"]<200 or c["oi"]<500 else ""
            rows+=f'<tr{atm}><td style="font-size:.6rem;color:var(--muted)">${c["strike"]:.0f}{tgt}</td><td style="color:var(--white);font-weight:700;font-size:.65rem">${c["price"]:.2f}</td><td style="color:var(--orange);font-size:.6rem">{c["iv"]}%</td><td style="color:{sc2};font-size:.6rem">${sp:.2f}</td><td style="color:var(--green-dim);font-size:.56rem">{c["volume"]//1000}K{liq}</td></tr>'
        return rows

    news_html=""
    for n in news:
        dc="#00ff9d" if n["sentiment"]=="bull" else ("#ff3a5e" if n["sentiment"]=="bear" else "#ffd700")
        sym="▲" if n["sentiment"]=="bull" else ("▼" if n["sentiment"]=="bear" else "—")
        news_html+=f'<div style="padding:4px 0;border-bottom:1px solid rgba(26,51,84,.35)"><div style="font-size:.64rem;line-height:1.3"><span style="color:{dc};font-weight:700;margin-right:3px">{sym}</span><a href="{n["link"]}" target="_blank" style="color:var(--text);text-decoration:none">{n["title"]}</a></div><div style="font-size:.55rem;color:var(--muted);margin-top:1px">{n["published"]}</div></div>'

    mtf_html=""
    for tfd in [d["daily"],d["h4"],d["h1"]]:
        if not tfd: continue
        if tfd["above_cloud"] and tfd["hh_hl"]: tr_,tc_="BULL","#00ff9d"
        elif tfd["below_cloud"]:                  tr_,tc_="BEAR","#ff3a5e"
        else:                                      tr_,tc_="NEUT","#ffd700"
        rl=f'{tfd["rsi"]}'; rc2="#00ff9d" if tfd["rsi_expansion_bull"] else ("#ff3a5e" if tfd["rsi_expansion_bear"] else "#ffd700")
        ml="↑" if tfd["macd_bull"] else "↓"; mc="#00ff9d" if tfd["macd_bull"] else "#ff3a5e"
        sc_=sum([tfd["above_cloud"],tfd["fast_bull"],tfd["macd_bull"],tfd["hh_hl"],tfd["rsi_expansion_bull"]])
        if sc_>=4: bias,bc2="LONG","#00ff9d"
        elif sc_<=1: bias,bc2="SHORT","#ff3a5e"
        else: bias,bc2="WATCH","#ffd700"
        vec="#00ff9d" if tfd["vol_expanded"] else ("#ffd700" if tfd["vol_ratio"]>=1.0 else "#ff3a5e")
        mtf_html+=f'<tr><td style="color:var(--white);font-weight:700;font-size:.62rem">{tfd["label"]}</td><td style="color:{tc_};font-weight:700;font-size:.62rem">{tr_}</td><td style="color:{rc2};font-size:.62rem">{rl}</td><td style="color:{mc};font-size:.62rem">{ml}{tfd["macd_h"]:+.2f}</td><td style="color:{bc2};font-weight:700;font-size:.62rem">{bias}</td><td style="color:{vec};font-size:.6rem">{tfd["vol_ratio"]}x</td></tr>'

    div_c=div["color"]; r_,g_,b_=int(div_c[1:3],16),int(div_c[3:5],16),int(div_c[5:7],16)
    if div["confirms_bull"] or div["confirms_bear"]: div_act=f'<span style="color:#ffd700;font-size:.6rem;font-weight:700">⚡ CONFIRMS ENTRY</span>'
    elif div["exit_warning_bull"] or div["exit_warning_bear"]: div_act=f'<span style="color:#ff3a5e;font-size:.6rem;font-weight:700">🚨 EXIT WARNING</span>'
    else: div_act=f'<span style="color:var(--muted);font-size:.58rem">No action</span>'

    on_banner=""
    if bctx["on_level"]:
        nl=bctx["nearest"]
        on_banner=f'<div style="background:rgba(255,215,0,.1);border:1px solid #ffd700;padding:5px 14px;font-size:.68rem;color:#ffd700;font-weight:700;text-align:center">⚡ ON LEVEL: ${nl[0]:.2f} — {nl[2]} ({TF_LABEL.get(nl[1],nl[1])})</div>'

    candle_rows="".join(f'<tr><td style="color:var(--muted);font-size:.58rem">{c["time"]}</td><td style="color:{"#00ff9d" if c["bull"] else "#ff3a5e"};font-weight:700;font-size:.64rem">${c["close"]:.2f}</td><td style="color:{"#00ff9d" if c["bull"] else "#ff3a5e"};font-size:.62rem">{"▲" if c["bull"] else "▼"}</td><td style="color:var(--muted);font-size:.58rem">{c["high"]:.1f}</td><td style="color:var(--muted);font-size:.58rem">{c["low"]:.1f}</td><td style="color:var(--green-dim);font-size:.56rem">{c["volume"]//1000}K</td></tr>' for c in d["recent_candles"]) or "<tr><td colspan='6' style='color:var(--muted);font-size:.6rem'>No data</td></tr>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="300">
<title>NVDA · OCC v6</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap" rel="stylesheet">
<style>
:root{{--bg:#06090f;--s1:#0b1420;--s2:#0f1a2e;--bd:#192e4a;
  --green:#00ff9d;--green-dim:#00c87a;--red:#ff3a5e;--yellow:#ffd700;
  --orange:#ff8c42;--blue:#4fc3f7;--white:#e8f4ff;--muted:#4a6888;--text:#b0c8e4;}}
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:var(--bg);color:var(--text);font-family:'Space Mono',monospace;overflow-x:hidden;
  background-image:radial-gradient(ellipse 60% 35% at 50% 0%,rgba(0,70,160,.09) 0%,transparent 65%)}}
.hdr{{border-bottom:1px solid var(--bd);padding:7px 14px;display:flex;align-items:center;justify-content:space-between;
  background:rgba(6,9,15,.97);backdrop-filter:blur(12px);position:sticky;top:0;z-index:100;flex-wrap:wrap;gap:5px}}
.main{{max-width:1440px;margin:0 auto;padding:6px 8px;display:flex;flex-direction:column;gap:5px}}
.g4{{display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:5px;align-items:start}}
.card{{background:var(--s1);border:1px solid var(--bd);padding:7px 9px}}
.ct{{font-size:.56rem;letter-spacing:1.8px;text-transform:uppercase;color:var(--muted);margin-bottom:6px;padding-bottom:5px;border-bottom:1px solid var(--bd);display:flex;align-items:center;gap:3px}}
.ct::before{{content:'';width:3px;height:3px;background:var(--blue);display:block;border-radius:50%;flex-shrink:0}}
table{{width:100%;border-collapse:collapse;table-layout:fixed}}
th,td{{padding:3px 3px;font-size:.64rem;border-bottom:1px solid rgba(25,46,74,.45);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
th{{color:var(--muted);font-size:.55rem;letter-spacing:.5px;text-align:left}}
tr:last-child td{{border-bottom:none}}
@keyframes pulse{{0%,100%{{opacity:1}}50%{{opacity:.3}}}}
.dot{{width:5px;height:5px;border-radius:50%;display:inline-block;animation:pulse 1.5s infinite}}
@media(max-width:1100px){{.g4{{grid-template-columns:1fr 1fr}}}}
@media(max-width:600px){{.g4{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<header class="hdr">
  <div style="display:flex;align-items:baseline;gap:7px;flex-wrap:wrap">
    <span style="font-family:'Syne',sans-serif;font-size:1.35rem;font-weight:800;color:var(--white);letter-spacing:3px">NVDA</span>
    <span style="font-size:1.1rem;color:{cc};font-weight:700">${price:.2f}</span>
    <span style="font-size:.78rem;color:{cc}">{arrow} {chg:+.2f} ({chgp:+.2f}%)</span>
    <span style="font-size:.58rem;color:var(--muted)">H{d["id_high"]}·L{d["id_low"]}·ATR${d["atr"]}</span>
    <span style="font-size:.55rem;background:rgba(79,195,247,.1);border:1px solid rgba(79,195,247,.3);color:#4fc3f7;padding:2px 6px;font-weight:700">v6 PROVEN</span>
  </div>
  <div style="display:flex;align-items:center;gap:7px;flex-wrap:wrap">
    <span style="font-size:.66rem;color:{bctx['scen_color']};font-weight:700">{bctx['scen_label']}</span>
    <span style="font-size:.64rem;color:{session['color']}"><span class="dot" style="background:{session['color']};margin-right:2px"></span>{session['emoji']} {session['name']}</span>
    <span style="color:{sc};font-size:.62rem;font-weight:700">SPY {sl}</span>
    <span style="font-size:.6rem;color:var(--muted)">{ds}·{ts}</span>
  </div>
</header>

<div class="main">
{on_banner}

<!-- SESSION BAR -->
<div style="display:grid;grid-template-columns:repeat(6,1fr);gap:3px">{tl}</div>

<!-- VERDICT -->
<div style="padding:9px 13px;border:1px solid var(--bd);border-left:4px solid {v['color']};background:linear-gradient(135deg,rgba(11,20,32,.97),rgba(15,26,46,.97));display:grid;grid-template-columns:1fr 140px;gap:10px;align-items:center">
  <div>
    <div style="font-size:.53rem;color:var(--muted);letter-spacing:1px;margin-bottom:2px">VERDICT · {ts}</div>
    <div style="font-family:'Syne',sans-serif;font-size:1.05rem;font-weight:800;color:{v['color']};line-height:1.2">{v['verdict']}</div>
    <div style="font-size:.66rem;color:var(--text);margin-top:3px;line-height:1.45">{v['explanation']}</div>
    <div style="margin-top:4px;padding:5px 7px;background:rgba(255,255,255,.025);border-left:2px solid {v['color']};font-size:.68rem;color:var(--white)">{v['trade_idea']}</div>
  </div>
  <div style="display:flex;flex-direction:column;gap:4px">
    <div style="border:1px solid {v['bias_color']};color:{v['bias_color']};padding:4px 8px;font-size:.7rem;letter-spacing:1px;font-weight:700;text-align:center">{v['bias']}</div>
    <div>
      <div style="display:flex;justify-content:space-between;font-size:.55rem;margin-bottom:2px"><span style="color:var(--muted)">CALL</span><span style="color:#00ff9d;font-weight:700">{al['call_count']}/{ms}</span></div>
      <div style="height:3px;background:rgba(255,255,255,.05);border-radius:2px"><div style="height:100%;width:{round(al['call_count']/ms*100)}%;background:linear-gradient(90deg,#00c87a,#00ff9d);border-radius:2px"></div></div>
    </div>
    <div>
      <div style="display:flex;justify-content:space-between;font-size:.55rem;margin-bottom:2px"><span style="color:var(--muted)">PUT</span><span style="color:#ff3a5e;font-weight:700">{al['put_count']}/{ms}</span></div>
      <div style="height:3px;background:rgba(255,255,255,.05);border-radius:2px"><div style="height:100%;width:{round(al['put_count']/ms*100)}%;background:linear-gradient(90deg,#cc1f40,#ff3a5e);border-radius:2px"></div></div>
    </div>
    <div style="font-size:.54rem;color:var(--muted);text-align:center;border-top:1px solid var(--bd);padding-top:3px">IV{opts['iv_30']}%·DTE{opts['dte']}·±${opts['exp_move']}</div>
  </div>
</div>

<!-- MAIN GRID -->
<div class="g4">

  <!-- COL 1: Signals + Divergence -->
  <div style="display:flex;flex-direction:column;gap:4px">
    <div style="font-size:.55rem;color:var(--muted);letter-spacing:1.5px;padding:2px 0">PROVEN SIGNALS</div>
    {sig_card("S1","ICHIMOKU",s1,"PF4.18","15M","#b39ddb")}
    {sig_card("S2","KAUFMAN", s2,"PF4.34","4H", "#7c4dff")}
    {sig_card("S3","RSI",     s3,"77%WR", "1H", "#4fc3f7")}
    {sig_card("S4","MACD+RSI",s4,"68%WR", "1H", "#ff8c42")}
    <div style="background:rgba({r_},{g_},{b_},.06);border:1px solid rgba({r_},{g_},{b_},.3);padding:6px 7px">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:2px">
        <span style="font-size:.55rem;color:var(--muted);letter-spacing:.5px">DIV BONUS</span>{div_act}
      </div>
      <div style="font-size:.7rem;font-weight:700;color:{div_c}">{div['label']}</div>
      <div style="font-size:.58rem;color:var(--text);margin-top:1px">{div['detail'] or "No divergence on 1H"}</div>
    </div>
  </div>

  <!-- COL 2: Brando + Scenario -->
  <div style="display:flex;flex-direction:column;gap:4px">
    <div class="card" style="padding:7px">
      <div class="ct">Brando S/R</div>
      <table style="table-layout:fixed">
        <colgroup><col style="width:20px"><col style="width:38px"><col><col style="width:32px"></colgroup>
        <tr><th>TF</th><th>$</th><th>LABEL</th><th style="text-align:right">DIST</th></tr>
        {brando_mini()}
      </table>
    </div>
    <div class="card" style="padding:7px">
      <div class="ct">Scenario</div>
      <div style="background:{bctx['scen_color']}10;border:1px solid {bctx['scen_color']}55;padding:5px;margin-bottom:5px">
        <div style="font-size:.73rem;font-weight:700;color:{bctx['scen_color']}">{bctx['scen_label']}</div>
        <div style="font-size:.55rem;color:var(--muted);margin-top:2px">R <strong style="color:#ff6b6b">${bctx['key_res'][0]:.2f}</strong>+{round(bctx['key_res'][0]-price,1)} · S <strong style="color:#00ff9d">${bctx['key_sup'][0]:.2f}</strong>-{round(price-bctx['key_sup'][0],1)}</div>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:4px;margin-bottom:4px">
        <div style="background:rgba(0,255,157,.04);border:1px solid rgba(0,255,157,.22);padding:4px">
          <div style="font-size:.52rem;color:#00ff9d;font-weight:700;margin-bottom:2px">A BULL</div>
          <div style="font-size:.6rem;color:var(--text);line-height:1.35">{bctx['scen_a']}</div>
        </div>
        <div style="background:rgba(255,58,94,.04);border:1px solid rgba(255,58,94,.22);padding:4px">
          <div style="font-size:.52rem;color:#ff3a5e;font-weight:700;margin-bottom:2px">B BEAR</div>
          <div style="font-size:.6rem;color:var(--text);line-height:1.35">{bctx['scen_b']}</div>
        </div>
      </div>
      <div style="font-size:.6rem;line-height:1.6;background:rgba(0,0,0,.15);padding:4px;border:1px solid var(--bd)">
        {"✅" if price>=191 else "❌"} <strong style="color:#ffd700">$191</strong>
        {"✅" if price>=184.58 else "❌"} <strong style="color:#4fc3f7">$184</strong>
        {"✅" if price>=180.34 else "❌"} <strong style="color:#ff8c42">$180★</strong>
        {"✅" if price>=175 else "❌"} <strong style="color:#7c4dff">$175</strong>
        {"✅" if price>=153.37 else "❌"} <strong style="color:#b39ddb">$153</strong>
      </div>
    </div>
  </div>

  <!-- COL 3: Call + Put -->
  <div style="display:grid;grid-template-rows:auto auto;gap:4px">
    <div class="card" style="padding:7px">
      <div class="ct">Call ({al['call_count']}/{ms})</div>
      {sig_col(al['call_reasons'],al['call_context'],al['call_count'],ms,"call")}
      {rr_mini(al['call_rr'],"call")}
    </div>
    <div class="card" style="padding:7px">
      <div class="ct">Put ({al['put_count']}/{ms})</div>
      {sig_col(al['put_reasons'],al['put_context'],al['put_count'],ms,"put")}
      {rr_mini(al['put_rr'],"put")}
    </div>
  </div>

  <!-- COL 4: Chain + News -->
  <div style="display:flex;flex-direction:column;gap:4px">
    <div class="card" style="padding:7px">
      <div class="ct">Chain {opts['expiry']} ({opts['dte']}d) IV{opts['iv_30']}%</div>
      <div style="font-size:.55rem;color:var(--orange);letter-spacing:1px;margin-bottom:2px">CALLS</div>
      <table style="table-layout:fixed">
        <colgroup><col style="width:36px"><col style="width:34px"><col style="width:28px"><col style="width:28px"><col></colgroup>
        <tr><th>STR</th><th>$</th><th>IV</th><th>SPR</th><th>VOL</th></tr>
        {chain_rows(opts['calls']) or "<tr><td colspan='5' style='color:var(--muted)'>—</td></tr>"}
      </table>
      <div style="font-size:.55rem;color:var(--red);letter-spacing:1px;margin:4px 0 2px">PUTS</div>
      <table style="table-layout:fixed">
        <colgroup><col style="width:36px"><col style="width:34px"><col style="width:28px"><col style="width:28px"><col></colgroup>
        <tr><th>STR</th><th>$</th><th>IV</th><th>SPR</th><th>VOL</th></tr>
        {chain_rows(opts['puts']) or "<tr><td colspan='5' style='color:var(--muted)'>—</td></tr>"}
      </table>
      <div style="margin-top:4px;font-size:.54rem;color:var(--muted)">🎯$3.5–6.5·V>500·OI>1K·Spr&lt;.20</div>
    </div>
    <div class="card" style="padding:7px">
      <div class="ct">News</div>
      {news_html or '<div style="font-size:.63rem;color:var(--muted)">No articles.</div>'}
    </div>
  </div>

</div>

<!-- BOTTOM GRID -->
<div class="g4">

  <div class="card" style="padding:7px">
    <div class="ct">MTF Bias</div>
    <table>
      <tr><th>TF</th><th>TREND</th><th>RSI</th><th>MACD</th><th>BIAS</th><th>VOL</th></tr>
      {mtf_html}
    </table>
    <div style="margin-top:4px;padding:4px;background:rgba(0,0,0,.15);border:1px solid var(--bd);font-size:.56rem;color:var(--muted)"><strong style="color:var(--yellow)">Size:</strong> 2=$500·3=$750·4=$1000</div>
  </div>

  <div class="card" style="padding:7px">
    <div class="ct">Regime</div>
    <div style="background:rgba(0,0,0,.2);border:1px solid {sc}55;padding:5px;margin-bottom:5px">
      <div style="font-size:.7rem;font-weight:700;color:{sc}">SPY {sl}</div>
      <div style="font-size:.56rem;color:var(--muted)">${d['spy_price']} vs 50EMA ${d['spy_e50']}</div>
    </div>
    <div style="margin-bottom:4px"><div style="display:flex;justify-content:space-between;font-size:.58rem;margin-bottom:2px"><span>Compression</span><span style="color:{cc2};font-weight:700">{comp}/100</span></div><div style="height:3px;background:rgba(255,255,255,.05);border-radius:2px"><div style="height:100%;width:{comp}%;background:{cc2};border-radius:2px"></div></div></div>
    <div style="display:flex;justify-content:space-between;font-size:.58rem;margin-bottom:3px"><span>ATR(14)</span><span style="color:var(--orange);font-weight:700">${d['atr']:.2f}</span></div>
    <div style="display:flex;justify-content:space-between;font-size:.58rem;margin-bottom:2px"><span>Volume</span><span style="font-weight:700;color:{vc}">{d['vol_ratio']}x·{vm:.0f}M</span></div>
    <div style="height:3px;background:rgba(255,255,255,.05);border-radius:2px"><div style="height:100%;width:{min(d['vol_ratio']*50,100):.0f}%;background:{vc};border-radius:2px"></div></div>
    <div style="margin-top:5px;border-top:1px solid var(--bd);padding-top:4px"><div style="font-size:.55rem;color:var(--muted);margin-bottom:2px">1H EMA</div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:2px;font-size:.6rem">
      <div>E8 <strong style="color:var(--blue)">${d['h1']['e8'] if d['h1'] else '—'}</strong></div>
      <div>E21 <strong style="color:var(--blue)">${d['h1']['e21'] if d['h1'] else '—'}</strong></div>
      <div>E50 <strong style="color:var(--blue)">${d['h1']['e50'] if d['h1'] else '—'}</strong></div>
      <div>RSI <strong style="color:var(--blue)">{d['h1']['rsi'] if d['h1'] else '—'}</strong></div>
    </div></div>
  </div>

  <div class="card" style="padding:7px">
    <div class="ct">5m Candles</div>
    <table>
      <tr><th>TIME</th><th>$</th><th></th><th>H</th><th>L</th><th>VOL</th></tr>
      {candle_rows}
    </table>
    <div style="margin-top:5px;border-top:1px solid var(--bd);padding-top:4px">
      <div style="font-size:.55rem;color:var(--muted);margin-bottom:2px">52W ({yr:.0f}th pct)</div>
      <div style="display:flex;justify-content:space-between;font-size:.56rem;color:var(--muted);margin-bottom:2px"><span>${d['week52l']:.0f}</span><span style="color:var(--white)">${price:.2f}</span><span>${d['week52h']:.0f}</span></div>
      <div style="height:3px;background:rgba(255,255,255,.05);border-radius:2px"><div style="height:100%;width:{yr:.0f}%;background:var(--blue);border-radius:2px"></div></div>
    </div>
  </div>

  <div class="card" style="padding:7px">
    <div class="ct">v6 Entry Rules</div>
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:3px;margin-bottom:5px">
      <div style="background:rgba(179,157,219,.08);border:1px solid rgba(179,157,219,.3);padding:3px 4px;text-align:center"><div style="font-size:.5rem;color:#b39ddb">ICHIMOKU</div><div style="font-size:.68rem;color:#00ff9d;font-weight:700">PF 4.18</div></div>
      <div style="background:rgba(124,77,255,.08);border:1px solid rgba(124,77,255,.3);padding:3px 4px;text-align:center"><div style="font-size:.5rem;color:#7c4dff">KAUFMAN</div><div style="font-size:.68rem;color:#00ff9d;font-weight:700">PF 4.34</div></div>
      <div style="background:rgba(79,195,247,.08);border:1px solid rgba(79,195,247,.3);padding:3px 4px;text-align:center"><div style="font-size:.5rem;color:#4fc3f7">RSI</div><div style="font-size:.68rem;color:#00ff9d;font-weight:700">77%WR</div></div>
    </div>
    <div style="font-size:.6rem;line-height:1.6;color:var(--text)">
      <span style="color:var(--green)">ENTRY:</span> 2+sigs·Brando·SPY·vol·Power Win<br>
      <span style="color:var(--yellow)">SIZE:</span> 2=$500·3=$750·4=$1000
    </div>
    <div style="margin-top:4px;padding:4px 6px;background:rgba(255,140,66,.07);border:1px solid rgba(255,140,66,.3);font-size:.58rem;color:var(--orange);line-height:1.5">EXIT +60%→SELL·-35%→STOP·Day3→OUT·MACD→HALF</div>
    <div style="margin-top:3px;padding:4px 6px;background:rgba(255,58,94,.05);border:1px solid rgba(255,58,94,.2);font-size:.58rem;color:var(--red);line-height:1.5">NEVER: midday 10:30–1PM·chasing·ignoring vol</div>
  </div>

</div>

<div style="text-align:center;padding:5px;font-size:.54rem;color:var(--muted);border-top:1px solid var(--bd)">
  v6·Ichimoku PF4.18·Kaufman PF4.34·RSI 77%WR·@EliteOptions2·⚠️Educational only·Updated <strong style="color:var(--text)">{ct_now.strftime("%Y-%m-%d %H:%M CT")}</strong>
</div>
</div>
</body></html>"""


if __name__ == "__main__":
    ct_now=datetime.datetime.now(CT); session=get_session(ct_now)
    print(f"⏰ {ct_now.strftime('%I:%M %p CT')} — {session['name']}")
    print("📡 Fetching...")
    d=fetch_all()
    print(f"   ${d['price']} | {d['change_pct']:+.2f}% | Vol {d['vol_ratio']}x | ATR ${d['atr']}")
    print(f"\n🔬 SIGNALS:")
    print(f"   S1 Ichi: {d['sig_ichimoku']['label']} — {d['sig_ichimoku']['detail']}")
    print(f"   S2 KAMA: {d['sig_kama']['label']} — {d['sig_kama']['detail']}")
    print(f"   S3 RSI:  {d['sig_rsi']['label']} — {d['sig_rsi']['detail']}")
    print(f"   S4 MACD: {d['sig_macd_combo']['label']} — {d['sig_macd_combo']['detail']}")
    print(f"   Div:     {d['h1_div']['label']}")
    bctx=get_brando_context(d["price"]); print(f"\n🗺️  {bctx['scen_label']}")
    opts=fetch_options(d["price"]); print(f"📋 {opts['expiry']} | {opts['dte']}d | IV{opts['iv_30']}%")
    news=fetch_news(); print(f"📰 {len(news)} articles")
    al=build_signals(d,opts,session,bctx); print(f"🧠 Call {al['call_count']}/{al['max_signals']} | Put {al['put_count']}/{al['max_signals']}")
    verdict=get_verdict(al,d,session,opts,bctx); print(f"   {verdict['verdict']}")
    html=render(d,opts,news,al,verdict,session,bctx,ct_now)
    Path("index.html").write_text(html,encoding="utf-8")
    print(f"✅ {len(html):,} bytes")
