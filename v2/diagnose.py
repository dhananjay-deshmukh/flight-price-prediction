import pandas as pd, numpy as np
tr = pd.read_csv("data/train.csv"); te = pd.read_csv("data/test.csv")

print("Missing values:\n", tr.isna().sum()[tr.isna().sum() > 0])
print("\nPrice describe:\n", tr.price.describe().round(1))
print("Price skew:", round(tr.price.skew(), 2), "| log-price skew:", round(np.log1p(tr.price).skew(), 2))
print("\nMean price by class:\n", tr.groupby("flight_class").price.agg(["mean", "count"]).round(0))

# Does id leak? (correlation with price, and with price inside each class)
print("\nCorr(id, price):", round(tr[["id", "price"]].corr().iloc[0, 1], 3))
print("Corr(id, price) within class:", tr.groupby("flight_class").apply(lambda g: round(g.id.corr(g.price), 3)).to_dict())
print("id range train:", tr.id.min(), "-", tr.id.max(), "| test:", te.id.min(), "-", te.id.max())

# How much do test flights overlap with train flights?
print("\nUnique flights train:", tr.flight.nunique(), "| test:", te.flight.nunique())
print("Test rows whose flight appears in train:", round(te.flight.isin(tr.flight).mean(), 3))
key = ["flight", "flight_class", "days_left"]
print("Test rows with exact (flight, class, days_left) match in train:",
      round(te.set_index(key).index.isin(tr.set_index(key).index).mean(), 3))

# Noise check: same flight+class+days_left, how much does price vary?
g = tr.groupby(key).price
print("Groups with >1 row:", (g.size() > 1).sum(), "| median within-group std:", round(g.std().median(), 1))

for c in ["airline", "stops", "source_city", "departure_time"]:
    print(f"\n{c}:", tr[c].value_counts().head(8).to_dict())
