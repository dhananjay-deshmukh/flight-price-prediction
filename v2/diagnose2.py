import pandas as pd, numpy as np
tr = pd.read_csv("data/train.csv").sort_values("id").reset_index(drop=True)

# 1) What is the data sorted by? Correlation of id with each column
for c in ["days_left", "duration"]:
    print(f"corr(id, {c}): {tr.id.corr(tr[c]):.3f}")
print("\nFirst 15 rows by id:\n", tr.head(15)[["id","airline","flight","source_city","destination_city","flight_class","days_left","price"]])

# 2) Do neighbouring ids have similar prices? (lag correlation)
print("\ncorr(price, price of previous id):", round(tr.price.corr(tr.price.shift(1)), 3))
lp = np.log1p(tr.price)
print("corr(log price, log price of previous id):", round(lp.corr(lp.shift(1)), 3))

# 3) Extreme prices
print("\nRows with price == 0:", (tr.price == 0).sum())
print("Rows with price > 2000:", (tr.price > 2000).sum())
print(tr.nlargest(8, "price")[["id","airline","flight","flight_class","stops","duration","days_left","price"]])
