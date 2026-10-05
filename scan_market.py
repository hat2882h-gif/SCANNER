# pip install yfinance pandas numpy
# يفحص كل أسهم ناسداك/NYSE بين 1 و5 دولار ويطلع market.csv (دورة الدعم + الارتكاز)
import io, urllib.request
import json, time
from concurrent.futures import ThreadPoolExecutor
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

import ftplib
def load_ib():
    """الشورت المتاح من FTP العام لـ Interactive Brokers (نفس مصدر iBorrowDesk، يتحدث كل 15 دقيقة)"""
    try:
        ftp = ftplib.FTP("ftp2.interactivebrokers.com", timeout=60)
        ftp.login("shortstock", "")
        files = ftp.nlst()
        print("ملفات IB:", files[:15])
        name = next((f for f in files if f.lower().startswith("usa")), None)
        if not name:
            print("ما لقيت ملف usa في IB")
            return {}
        buf = io.BytesIO()
        ftp.retrbinary("RETR " + name, buf.write)
        ftp.quit()
        out, idx = {}, 7
        for line in buf.getvalue().decode("utf-8", "ignore").splitlines():
            if line.startswith("#SYM"):
                cols = line.lstrip("#").split("|")
                idx = cols.index("AVAILABLE") if "AVAILABLE" in cols else 7
                continue
            if not line or line.startswith("#"):
                continue
            c = line.split("|")
            if len(c) <= idx:
                continue
            m = re.sub(r"[^\d]", "", c[idx])
            if m:
                out[c[0].strip()] = int(m)
        print("IB: عدد الأسهم في الملف", len(out))
        return out
    except Exception as e:
        print("IB FTP فشل:", e)
        return {}

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
            macd_pos = int(dif.iloc[-1] > 0)
            op, hi, lo = df["Open"].values, df["High"].values, df["Low"].values
            gap = 0.0
            for k in range(1, len(df)):
                if op[k] < lo[k - 1] * 0.98 and hi[k:].max() < lo[k - 1]:
                    gap = max(gap, (lo[k - 1] - price) / price * 100)
            l20, h20 = df["Low"].tail(20).values, df["High"].tail(20).values
            rise20 = float(((h20 - np.minimum.accumulate(l20)) / np.minimum.accumulate(l20)).max() * 100)
            rows.append(dict(symbol=s, name=names[s], country="أمريكا", price=round(price, 3), support=round(sup, 3),
                             stable_days=stable, rebound=round(reb, 1), tested=tested, confirmed=confirmed,
                             rsi=round(float(rs.iloc[-1]), 1), split_ratio=split_ratio, split_date=split_date,
                             dd=round(dd, 1), ema_ok=ema_ok, rsi_min=round(rsi_min, 1), macd_x=macd_x,
                             gap=round(gap, 1), short_avail=-1, news_n=0, rise20=round(rise20, 1), macd_pos=macd_pos, short_src=""))
        except Exception:
            pass

# الشورت والأخبار: فقط للأسهم اللي حققت الشروط الفنية الخمسة (توفيرًا للوقت)
# الشورت المتاح من IB (ملف واحد لكل السوق). السهم غير الموجود بالقائمة = غير قابل للشورت = 0
ib = load_ib()
ib_ok = len(ib) > 1000
for r in rows:
    if ib_ok:
        r["short_avail"], r["short_src"] = ib.get(r["symbol"], 0), "ib"
print("الشورت من IB:", "نجح" if ib_ok else "فشل", "| أسهم بقيمة:", sum(r["short_src"] == "ib" for r in rows))

for r in rows:
    tech = int(r["dd"] >= DD_MIN) + r["ema_ok"] + int(r["rsi_min"] <= RSI_MAX) + r["macd_x"] + int(r["gap"] >= GAP_MIN)
    r["anchor_n"] = tech
    if tech == 5:
        try:
            t = yf.Ticker(r["symbol"])
            r["news_n"] = len(t.news or [])
        except Exception:
            pass
        r["anchor_n"] += int(0 <= r["short_avail"] < SHORT_MAX) + int(r["news_n"] >= 1)

cols = ["symbol","name","country","price","support","stable_days","rebound","tested","confirmed","rsi","split_ratio","split_date","dd","ema_ok","rsi_min","macd_x","gap","short_avail","news_n","anchor_n","rise20","macd_pos","short_src"]
pd.DataFrame(rows)[cols].to_csv("market.csv", index=False)
print("تم حفظ market.csv:", len(rows), "سهم | ارتكاز كامل 7/7:", sum(r["anchor_n"] == 7 for r in rows))

