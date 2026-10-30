import warnings; warnings.filterwarnings("ignore")
exec(open("v2/step3_compare.py").read().split('run("A')[0])   # reuse tr, y, ly, Xb, params, kf, neighbour_feats
tr["route"] = tr.source_city + "_" + tr.destination_city

def nn_feats(q, ref, keys, tag, exclude_self, k=1):
    """Log-price and id-gap of the k nearest lower/higher ids in ref, optionally within the same group."""
    cols = [f"{tag}_{s}{d}_{f}" for s in ("prev", "next") for d in range(1, k+1) for f in ("lp", "gap")]
    out = pd.DataFrame(np.nan, index=q.index, columns=cols)
    if keys is None:
        pairs = [(q, ref)]
    else:
        rg = {kk: g for kk, g in ref.groupby(keys)}
        pairs = [(qg, rg.get(kk)) for kk, qg in q.groupby(keys)]
    for qg, g in pairs:
        if g is None or len(g) == 0: continue
        g = g.sort_values("id"); rid = g.id.values; rlp = np.log1p(g.price.values); qid = qg.id.values
        pos = np.searchsorted(rid, qid); nxt = pos.copy()
        if exclude_self:
            hit = (pos < len(rid)) & (rid[np.minimum(pos, len(rid)-1)] == qid); nxt = pos + hit
        for d in range(1, k+1):
            for s, idx in (("prev", pos - d), ("next", nxt + d - 1)):
                ok = (idx >= 0) & (idx < len(rid)); i = np.clip(idx, 0, len(rid)-1)
                out.loc[qg.index, f"{tag}_{s}{d}_lp"]  = np.where(ok, rlp[i], np.nan)
                out.loc[qg.index, f"{tag}_{s}{d}_gap"] = np.where(ok, np.abs(rid[i] - qid), np.nan)
    return out

def rich_nn(q, ref, ex):
    return pd.concat([nn_feats(q, ref, None, "all", ex, k=3),
                      nn_feats(q, ref, ["route", "flight_class"], "rc", ex, k=2),
                      nn_feats(q, ref, ["flight", "flight_class"], "fl", ex, k=1)], axis=1)

def run3(name, feat_fn):
    s_all, s_clip = [], []
    for a, b in kf.split(Xb):
        Xa = Xb.iloc[a].join(feat_fn(tr.iloc[a], tr.iloc[a], True))
        Xv = Xb.iloc[b].join(feat_fn(tr.iloc[b], tr.iloc[a], False))
        p = np.expm1(lgb.LGBMRegressor(**params).fit(Xa, ly[a]).predict(Xv))
        s_all.append(r2_score(y[b], p)); keep = y[b] <= 2000
        s_clip.append(r2_score(y[b][keep], p[keep]))
    print(f"{name:35s} R2 {np.mean(s_all):.4f} | folds {np.round(s_all,3)} | w/o >$2000 {np.mean(s_clip):.4f}", flush=True)

run3("D: 1 neighbour each side", neighbour_feats)
run3("G: rich neighbours (all+route+flight)", rich_nn)
