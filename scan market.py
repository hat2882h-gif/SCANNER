# pip install yfinance pandas
# يفحص كل أسهم ناسداك/NYSE بين 1 و5 دولار ويطلع ملف market.csv جاهز للتطبيق
import io, urllib.request
import pandas as pd, numpy as np, yfinance as yf

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

# 1) فلترة السعر بين 1 و5
cands = []
for b in batches(syms, 500):
    d = yf.download(b, period="5d", progress=False, threads=True, auto_adjust=True)["Close"]
    last = d.ffill().iloc[-1]
    cands += list(last[(last >= 1) & (last <= 5)].index)
print("أسهم بين 1 و5 دولار:", len(cands))

def rsi(c, n=14):
    d = c.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    return float(100 - 100 / (1 + up.iloc[-1] / dn.iloc[-1])) if dn.iloc[-1] else 100.0

rows = []
for b in batches(cands, 100):
    h = yf.download(b, period="1y", progress=False, threads=True, group_by="ticker", auto_adjust=True, actions=True)
    for s in b:
        try:
            df = h[s].dropna()
            if len(df) < 40:
                continue
            w = df.tail(60).reset_index(drop=True)
            i = int(w["Low"].idxmin())          # القاع
            sup = float(w["Low"].iloc[i])
            after = w.iloc[i:]
            price = float(w["Close"].iloc[-1])
            stable = len(w) - 1 - i             # جلسات منذ القاع
            peak_i = int(after["High"].idxmax())
            reb = (float(after["High"].max()) - sup) / sup * 100
            post = w.iloc[peak_i:]
            tested = int(reb >= 10 and len(post) > 2 and float(post["Low"].min()) <= sup * 1.05
                         and float(post["Low"].min()) >= sup * 0.99)
            confirmed = int(tested and price > sup * 1.03)
            # التقسيم العكسي (reverse split): النسبة أقل من 1، مثل 0.125 = 1:8
            sp = df["Stock Splits"] if "Stock Splits" in df else pd.Series(dtype=float)
            sp = sp[(sp > 0) & (sp < 1)]
            split_ratio, split_date = "", ""
            if len(sp):
                split_ratio = "1:%d" % round(1 / float(sp.iloc[-1]))
                split_date = str(sp.index[-1].date())
            rows.append([s, names[s], "أمريكا", round(price, 3), round(sup, 3), stable,
                         round(reb, 1), tested, confirmed, round(rsi(w["Close"]), 1), split_ratio, split_date])
        except Exception:
            pass

out = pd.DataFrame(rows, columns="symbol,name,country,price,support,stable_days,rebound,tested,confirmed,rsi,split_ratio,split_date".split(","))
out.to_csv("market.csv", index=False)
print("تم حفظ market.csv:", len(out), "سهم")
