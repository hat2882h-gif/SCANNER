# pip install yfinance pandas numpy
# يفحص كل أسهم ناسداك/NYSE بين 1 و5 دولار ويطلع market.csv (دورة الدعم + الارتكاز)
import io, urllib.request
import json, time
import pandas as pd, numpy as np, yfinance as yf

# ===== حدود الارتكاز (غيّرها كما تريد) =====
DD_MIN = 50      # أدنى هبوط من أعلى سعر خلال سنة % (نماذج الهبوط مكتملة)
RSI_MAX = 30     # تشبع بيعي: أدنى RSI خلال آخر 5 جلسات
GAP_MIN = 30     # أدنى فجوة هبوطية غير مغلقة فوق السعر %
SHORT_MAX = 50000  # الشورت المتاح (أسهم) من iBorrowDesk: أقل من 50 ألف، وكل ما قل أفضل
# ==========================================

def get(url):
    return pd.read_csv(io.StringIO(urllib.request.urlopen(url).read().decode()), sep="|")

n = get("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt")
o = get("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt")
n = n.rename(columns={"Symbol": "sym"})[["sym", "Security Name", "Test Issue", "ETF"]]
o = o.rename(columns={"ACT Symbol": "sym"})[["sym", "Security Name", "Test Issue", "ETF"]]
u = pd.concat([n, o])
u = u[(u["Test Issue"] == "N") & (u["ETF"] == "N")].dropna(subset=["sym"])
u = u[~u.sym.str.contains(r"[\$\.\^]", regex=True, na=False)]
names = dict(zip(u.sym, u["Security Name"].str.replace(",", " ")))
syms = list(names)

def batches(l, k):
    for i in range(0, len(l), k):
        yield l[i:i + k]

cands = []
for b in batches(syms, 500):
    d = yf.download(b, period="5d", progress=False, threads=True, auto_adjust=True)["Close"]
    last = d.ffill().iloc[-1]
    cands += list(last[(last >= 1) & (last <= 5)].index)
print("أسهم بين 1 و5 دولار:", len(cands))

_dbg = [True]
def borrow_avail(sym):
    """الأسهم المتاحة للشورت من iBorrowDesk (API غير رسمي). -1 = غير متوفر"""
    try:
        req = urllib.request.Request("https://iborrowdesk.com/api/ticker/" + sym,
                                     headers={"User-Agent": "Mozilla/5.0"})
        raw = urllib.request.urlopen(req, timeout=20).read().decode()
        if _dbg[0]:
            print("iBorrowDesk نموذج:", raw[:300]); _dbg[0] = False
        d = json.loads(raw)
        items = d.get("real_time") or d.get("daily") or []
        if not items:
            return -1
        it = max(items, key=lambda x: str(x.get("time") or x.get("date") or ""))
        return int(it.get("available", -1))
    except Exception as e:
        if _dbg[0]:
            print("iBorrowDesk خطأ:", e); _dbg[0] = False
        return -1

def ema(s, k): return s.ewm(span=k, adjust=False).mean()

def rsi_series(c, k=14):
    d = c.diff()
    up = d.clip(lower=0).rolling(k).mean()
    dn = (-d.clip(upper=0)).rolling(k).mean()
    return (100 - 100 / (1 + up / dn.replace(0, np.nan))).fillna(100)

rows = []
for b in batches(cands, 100):
    h = yf.download(b, period="1y", progress=False, threads=True, group_by="ticker", auto_adjust=True, actions=True)
    for s in b:
        try:
            df = h[s].dropna(subset=["Close"])
            if len(df) < 60:
                continue
            c = df["Close"]
            w = df.tail(60).reset_index(drop=True)
            i = int(w["Low"].idxmin())
            sup = float(w["Low"].iloc[i])
            after = w.iloc[i:]
            price = float(c.iloc[-1])
            stable = len(w) - 1 - i
            peak_i = int(after["High"].idxmax())
            reb = (float(after["High"].max()) - sup) / sup * 100
            post = w.iloc[peak_i:]
            tested = int(reb >= 10 and len(post) > 2 and float(post["Low"].min()) <= sup * 1.05
                         and float(post["Low"].min()) >= sup * 0.99)
            confirmed = int(tested and price > sup * 1.03)
            rs = rsi_series(c)
            sp = df["Stock Splits"] if "Stock Splits" in df else pd.Series(dtype=float)
            sp = sp[(sp > 0) & (sp < 1)]
            split_ratio, split_date = "", ""
            if len(sp):
                split_ratio = "1:%d" % round(1 / float(sp.iloc[-1]))
                split_date = str(sp.index[-1].date())
            # ---- الارتكاز ----
            hi_y = float(df["High"].max())
            dd = (hi_y - price) / hi_y * 100
            ema_ok = int(price < float(ema(c, 30).iloc[-1]) and price < float(ema(c, 50).iloc[-1]))
            rsi_min = float(rs.tail(5).min())
            macd = ema(c, 12) - ema(c, 26)
            dif = (macd - ema(macd, 9)).tail(4)
            macd_x = int(dif.iloc[-1] > 0 and (dif.iloc[:-1] <= 0).any())
            op, hi, lo = df["Open"].values, df["High"].values, df["Low"].values
            gap = 0.0
            for k in range(1, len(df)):
                if op[k] < lo[k - 1] * 0.98 and hi[k:].max() < lo[k - 1]:
                    gap = max(gap, (lo[k - 1] - price) / price * 100)
            rows.append(dict(symbol=s, name=names[s], country="أمريكا", price=round(price, 3), support=round(sup, 3),
                             stable_days=stable, rebound=round(reb, 1), tested=tested, confirmed=confirmed,
                             rsi=round(float(rs.iloc[-1]), 1), split_ratio=split_ratio, split_date=split_date,
                             dd=round(dd, 1), ema_ok=ema_ok, rsi_min=round(rsi_min, 1), macd_x=macd_x,
                             gap=round(gap, 1), short_avail=-1, news_n=0))
        except Exception:
            pass

# الشورت والأخبار: فقط للأسهم اللي حققت الشروط الفنية الخمسة (توفيرًا للوقت)
for r in rows:
    tech = int(r["dd"] >= DD_MIN) + r["ema_ok"] + int(r["rsi_min"] <= RSI_MAX) + r["macd_x"] + int(r["gap"] >= GAP_MIN)
    r["anchor_n"] = tech
    if tech == 5:
        try:
            t = yf.Ticker(r["symbol"])
            r["news_n"] = len(t.news or [])
        except Exception:
            pass
        r["short_avail"] = borrow_avail(r["symbol"])
        time.sleep(0.5)
        r["anchor_n"] += int(0 <= r["short_avail"] < SHORT_MAX) + int(r["news_n"] >= 1)

pd.DataFrame(rows).to_csv("market.csv", index=False)
print("تم حفظ market.csv:", len(rows), "سهم | ارتكاز كامل 7/7:", sum(r["anchor_n"] == 7 for r in rows))

