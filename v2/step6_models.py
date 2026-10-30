import warnings; warnings.filterwarnings("ignore")
exec(open("v2/step3_compare.py").read().split('run("A')[0])   # tr, y, ly, Xb, params, kf, neighbour_feats
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from itertools import combinations

def codes(X):  # numeric version for Extra Trees
    X = X.copy()
    for c in X.select_dtypes("category"): X[c] = X[c].cat.codes
    return X.fillna(-1)

names = ["lgb", "lgb_clean", "cat", "et"]
oof = {n: np.zeros(len(tr)) for n in names}          # out-of-fold LOG predictions
folds = list(kf.split(Xb))
for a, b in folds:
    Xa = Xb.iloc[a].join(neighbour_feats(tr.iloc[a], tr.iloc[a], True))
    Xv = Xb.iloc[b].join(neighbour_feats(tr.iloc[b], tr.iloc[a], False))
    clean = (y[a] > 0) & (y[a] <= 3000)

    oof["lgb"][b] = lgb.LGBMRegressor(**params).fit(Xa, ly[a]).predict(Xv)
    oof["lgb_clean"][b] = lgb.LGBMRegressor(**params).fit(Xa[clean], ly[a][clean]).predict(Xv)
    cat_cols = list(Xa.select_dtypes("category").columns)
    Ca, Cv = Xa.copy(), Xv.copy()
    for c in cat_cols: Ca[c] = Ca[c].astype(str); Cv[c] = Cv[c].astype(str)
    oof["cat"][b] = CatBoostRegressor(iterations=2000, learning_rate=0.05, depth=8, verbose=0,
                                      random_seed=42, thread_count=-1).fit(Ca, ly[a], cat_features=cat_cols).predict(Cv)
    oof["et"][b] = ExtraTreesRegressor(400, min_samples_leaf=2, max_features=0.7, n_jobs=-1,
                                       random_state=42).fit(codes(Xa), ly[a]).predict(codes(Xv))
    print("fold done", flush=True)

def score(logp):
    p = np.expm1(logp); s_all, s_clip = [], []
    for a, b in folds:
        s_all.append(r2_score(y[b], p[b])); k = y[b] <= 2000
        s_clip.append(r2_score(y[b][k], p[b][k]))
    return np.mean(s_all), np.mean(s_clip)

print("\nSingle models:")
for n in names:
    s, c = score(oof[n]); print(f"  {n:12s} R2 {s:.4f} | w/o >$2000 {c:.4f}")
print("\nBlends (average of log predictions):")
res = []
for r in [2, 3, 4]:
    for combo in combinations(names, r):
        s, c = score(sum(oof[n] for n in combo) / r); res.append((c, s, combo))
for c, s, combo in sorted(res, reverse=True)[:6]:
    print(f"  {' + '.join(combo):30s} R2 {s:.4f} | w/o >$2000 {c:.4f}")
np.save("v2/oof_step6.npy", oof, allow_pickle=True)
