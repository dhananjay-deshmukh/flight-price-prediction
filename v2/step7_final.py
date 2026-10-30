import warnings; warnings.filterwarnings("ignore")
exec(open("v2/step6_models.py").read().split("names = ")[0])   # imports, base(), neighbour_feats, params, codes()
import os
te = pd.read_csv("data/test.csv")
sample = pd.read_csv("data/submission.csv")
OUT = "outputs/submissions_v2"; os.makedirs(OUT, exist_ok=True)

# Same category levels for train and test
full = pd.concat([tr.drop(columns="price"), te], ignore_index=True)
Xall = base(full)
Xtr = Xall.iloc[:len(tr)].reset_index(drop=True).join(neighbour_feats(tr, tr, True))
Xte = Xall.iloc[len(tr):].reset_index(drop=True).join(neighbour_feats(te, tr, False).reset_index(drop=True))

clean = (y > 0) & (y <= 3000)
print("Training LightGBM (clean)..."); p_lgb = lgb.LGBMRegressor(**params).fit(Xtr[clean], ly[clean]).predict(Xte)
print("Training CatBoost...")
cat_cols = list(Xtr.select_dtypes("category").columns)
Ctr, Cte = Xtr.copy(), Xte.copy()
for c in cat_cols: Ctr[c] = Ctr[c].astype(str); Cte[c] = Cte[c].astype(str)
p_cat = CatBoostRegressor(iterations=2000, learning_rate=0.05, depth=8, verbose=0, random_seed=42,
                          thread_count=-1).fit(Ctr, ly, cat_features=cat_cols).predict(Cte)
print("Training Extra Trees...")
p_et = ExtraTreesRegressor(400, min_samples_leaf=2, max_features=0.7, n_jobs=-1,
                           random_state=42).fit(codes(Xtr), ly).predict(codes(Xte))

def save(name, logp):
    sub = pd.DataFrame({"id": te["id"], "price": np.expm1(logp)})
    assert len(sub) == len(sample) and set(sub.id) == set(sample.id) and sub.price.notna().all()
    sub.to_csv(f"{OUT}/{name}.csv", index=False)
    print(f"Saved {name}.csv | mean {sub.price.mean():.1f} median {sub.price.median():.1f} min {sub.price.min():.1f} max {sub.price.max():.1f}")

save("blend_lgbclean_cat_et", (p_lgb + p_cat + p_et) / 3)
save("blend_lgbclean_cat", (p_lgb + p_cat) / 2)
save("catboost", p_cat)
print("\nTrain price: mean", round(y.mean(), 1), "median", np.median(y))
