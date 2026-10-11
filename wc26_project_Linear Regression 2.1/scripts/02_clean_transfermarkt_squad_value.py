"""
Clean raw R2 (Transfermarkt 'most valuable national teams', detailed view, pasted as text)
-> clean/R2_squad_value.csv (and a .xlsx copy)

Input : raw/R2_transfermarkt_raw.xlsx   columns: # | Country | Confederation | Squad size | ø age | Market Value | Ø market value of players
Steps : keep the 48 World Cup teams, convert '€1.56bn', '€946.00m', '€876k' to euros,
        standardise team names, take log of market value, validate, save.
"""
import re, numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # project root = folder above /scripts
(ROOT / "clean").mkdir(exist_ok=True)

# The 48 teams, spelled as in our project (taken from the cleaned FIFA ranking file)
teams = pd.read_csv(ROOT / "clean" / "R1_fifa_ranking.csv")["team"].tolist()
# Transfermarkt spells a few teams differently from our project standard
TM_NAME = {"Turkey": "Turkiye", "Czech Republic": "Czechia", "DR Congo": "Democratic Republic of the Congo",
           "Bosnia and Herzegovina": "Bosnia-Herzegovina"}

raw = pd.read_excel(ROOT / "raw" / "R2_transfermarkt_raw.xlsx")
raw.columns = ["rank", "country", "confederation", "squad_size", "avg_age", "market_value", "avg_player_value"]

def to_eur(s):                                          # '€1.56bn' -> 1560000000.0
    num, unit = re.fullmatch(r"€([\d.,]+)(bn|m|k)", str(s).strip()).groups()
    return float(num.replace(",", "")) * {"bn": 1e9, "m": 1e6, "k": 1e3}[unit]

raw["country"] = raw["country"].astype(str).str.strip()
tm_to_project = {v: k for k, v in TM_NAME.items()}
sel = raw[raw.country.isin([TM_NAME.get(t, t) for t in teams])].copy()
sel["team"] = sel.country.replace(tm_to_project)
missing = sorted(set(teams) - set(sel.team))
assert not missing, f"Teams not found in the Excel file: {missing}"

df = pd.DataFrame({
    "team": sel.team, "tm_name": sel.country, "squad_size": sel.squad_size.astype(int),
    "avg_age": sel.avg_age.astype(float),
    "market_value_eur": sel.market_value.map(to_eur),
    "avg_player_value_eur": sel.avg_player_value.map(to_eur),
}).reset_index(drop=True)
df["log_market_value"] = np.log(df.market_value_eur)

# ---- validation checks ----
assert len(df) == 48 and df.team.is_unique
assert df.avg_age.between(20, 35).all()
assert (df.market_value_eur > 1e6).all()
# consistency: total value should be roughly squad size x average player value
ratio = df.market_value_eur / (df.squad_size * df.avg_player_value_eur)
assert ratio.between(0.9, 1.1).all(), df.loc[~ratio.between(0.9, 1.1), "team"].tolist()

# cross-source check against FIFA ranking points (two independent sources should agree on who is strong)
r1 = pd.read_csv(ROOT / "clean" / "R1_fifa_ranking.csv")
chk = df.merge(r1, on="team")
print("Correlation: log(squad value) vs FIFA points =", round(chk.log_market_value.corr(chk.fifa_points), 2))

out = df[["team", "squad_size", "avg_age", "market_value_eur", "avg_player_value_eur", "log_market_value", "tm_name"]]
out.to_csv(ROOT / "clean" / "R2_squad_value.csv", index=False)
out.to_excel(ROOT / "clean" / "R2_squad_value.xlsx", index=False)
print(out.sort_values("market_value_eur", ascending=False).head(), "\nrows:", len(out))
