import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

tr = pd.read_csv("data/train.csv")
y = tr.price.values.astype(float); ly = np.log1p(y)
CAT = ["airline","flight","source_city","departure_time","stops",
       "arrival_time","destination_city","flight_class","route"]

def base(df):
    X = df.drop(columns=["price"], errors="ignore").copy()
    X["route"] = X.source_city + "_" + X.destination_city
    for c in CAT: X[c] = X[c].astype("category")
    return X

def neighbour_feats(q, ref, exclude_self):
    """For each query row, log-price of nearest lower/higher id in ref (ref = training rows only)."""
    ref = ref.sort_values("id"); rid = ref.id.values; rlp = np.log1p(ref.price.values)
    rkey = (ref.source_city + ref.destination_city + ref.flight_class).values
    qid = q.id.values; qkey = (q.source_city + q.destination_city + q.flight_class).values
    pos = np.searchsorted(rid, qid)
    nxt = pos.copy()
    if exclude_self:  # training rows: skip themselves
        hit = (pos < len(rid)) & (rid[np.minimum(pos, len(rid)-1)] == qid)
        nxt = pos + hit
    prv = pos - 1
    out = pd.DataFrame(index=q.index)
    for name, idx in [("prev", prv), ("next", nxt)]:
        ok = (idx >= 0) & (idx < len(rid)); i = np.clip(idx, 0, len(rid)-1)
        out[f"{name}_lp"]   = np.where(ok, rlp[i], np.nan)
        out[f"{name}_gap"]  = np.where(ok, np.abs(rid[i] - qid), np.nan)
        out[f"{name}_same"] = np.where(ok, (rkey[i] == qkey).astype(float), np.nan)
    return out

params = dict(n_estimators=1500, learning_rate=0.03, num_leaves=63,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
              min_child_samples=20, verbose=-1, random_state=42)
kf = KFold(5, shuffle=True, random_state=42)
Xb = base(tr)

def run(name, use_log, use_id, use_neigh):
    scores, oof = [], np.zeros(len(tr))
    for a, b in kf.split(Xb):
        Xa, Xv = Xb.iloc[a].copy(), Xb.iloc[b].copy()
        if not use_id: Xa, Xv = Xa.drop(columns="id"), Xv.drop(columns="id")
        if use_neigh:
            ref = tr.iloc[a]
            Xa = Xa.join(neighbour_feats(tr.iloc[a], ref, True))
            Xv = Xv.join(neighbour_feats(tr.iloc[b], ref, False))
        target = ly[a] if use_log else y[a]
        m = lgb.LGBMRegressor(**params).fit(Xa, target)
        p = m.predict(Xv); p = np.expm1(p) if use_log else p
        oof[b] = p; scores.append(r2_score(y[b], p))
    print(f"{name:45s} R2 {np.mean(scores):.4f} ± {np.std(scores):.4f}", flush=True)
    return oof

run("A: raw price, no id",              False, False, False)
run("B: log price, no id",              True,  False, False)
run("C: log price + id",                True,  True,  False)
run("D: log price + id + neighbour prices", True, True, True)
