import matplotlib.pyplot as plt
import pandas as pd
import math
import scipy.stats as st

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', 10)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

# GÖREV 1: Average Rating'i Güncel Yorumlara Göre Hesaplayınız ve Var Olan Average Rating ile Kıyaslayınız.

# Adım 1: Veri Setini Okutunuz ve Ürünün Ortalama Puanını Hesaplayınız.

df = pd.read_csv("datasets/amazon_review.csv")
df.head()
df["reviewerID"].nunique()
df.info()
df.isnull().sum()
df["overall"].mean()

# Adım 2: Tarihe Göre Ağırlıklı Puan Ortalamasını Hesaplayınız.

q1 = df["day_diff"].quantile(0.25)  # 281
q2 = df["day_diff"].quantile(0.50)  # 431
q3 = df["day_diff"].quantile(0.75)  # 601

weighted_average = df.loc[df["day_diff"] <= q1, "overall"].mean() * 50 / 100 + \
                   df.loc[(df["day_diff"] > q1) & (df["day_diff"] <= q2), "overall"].mean() * 25 / 100 + \
                   df.loc[(df["day_diff"] > q2) & (df["day_diff"] <= q3), "overall"].mean() * 15 / 100 + \
                   df.loc[df["day_diff"] > q3, "overall"].mean() * 10 / 100

# Adım 3: Ağırlıklandırılmış puanlamada her bir zaman diliminin ortalamasını karşılaştırıp yorumlayınız.
df.loc[df["day_diff"] <= q1, "overall"].mean()  # 4.70
df.loc[(df["day_diff"] > q1) & (df["day_diff"] <= q2), "overall"].mean()  # 4.63
df.loc[(df["day_diff"] > q2) & (df["day_diff"] <= q3), "overall"].mean()  # 4.57
df.loc[df["day_diff"] > q3, "overall"].mean()  # 4.44

# Görev 2: Ürün için Ürün Detay Sayfasında Görüntülenecek 20 Review'i Belirleyiniz.

# Adım 1. helpful_no Değişkenini Üretiniz
df["helpful_no"] = df["total_vote"] - df["helpful_yes"]

# Adım 2. score_pos_neg_diff, score_average_rating ve wilson_lower_bound Skorlarını Hesaplayıp Veriye Ekleyiniz


def score_up_down(up, down):
    return up - down

df["score_pos_neg_diff"] = df.apply(lambda x: score_up_down(x["helpful_yes"], x["helpful_no"]), axis=1)


def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up + down)

df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"], x["helpful_no"]), axis=1)


def wilson_lower_bound(up, down, confidence=0.95):
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)


df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)

# Adım 3. 20 Yorumu Belirleyiniz

df.sort_values("wilson_lower_bound", ascending=False).head(20)

