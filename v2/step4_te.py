import warnings; warnings.filterwarnings("ignore")
exec(open("v2/step3_compare.py").read().split("run(\"A")[0])   # reuse data, base(), neighbour_feats(), params, kf

def target_encode(fit_df, fit_t, apply_df, cols, k=10):
    """Smoothed mean of log-price per group, learned on fit_df only."""
    m = fit_t.mean()
    g = fit_df[cols].assign(t=fit_t).groupby(cols, observed=True).t.agg(["sum", "count"])
    enc = (g["sum"] + k * m) / (g["count"] + k)
    return apply_df[cols].merge(enc.rename("e"), left_on=cols, right_index=True, how="left")["e"].fillna(m).values

TE_SETS = [["flight", "flight_class"],
           ["flight", "flight_class", "days_left"],
           ["route", "flight_class", "days_left"],
           ["airline", "flight_class", "days_left"]]

def add_te(Xa, Xv, ya_log):
    Xa, Xv = Xa.copy(), Xv.copy()
    inner = KFold(5, shuffle=True, random_state=1)
    for cols in TE_SETS:
        name = "te_" + "_".join(cols)
        oof = np.zeros(len(Xa))
        for i, j in inner.split(Xa):        # out-of-fold for training rows -> no self-leak
            oof[j] = target_encode(Xa.iloc[i], ya_log[i], Xa.iloc[j], cols)
        Xa[name] = oof
        Xv[name] = target_encode(Xa, ya_log, Xv, cols)
    return Xa, Xv

def run2(name, use_te, use_neigh):
    s_all, s_clip = [], []
    for a, b in kf.split(Xb):
        Xa, Xv = Xb.iloc[a].drop(columns="id"), Xb.iloc[b].drop(columns="id")
        if use_te: Xa, Xv = add_te(Xa, Xv, ly[a])
        if use_neigh:
            Xa = Xa.join(neighbour_feats(tr.iloc[a], tr.iloc[a], True))
            Xv = Xv.join(neighbour_feats(tr.iloc[b], tr.iloc[a], False))
        p = np.expm1(lgb.LGBMRegressor(**params).fit(Xa, ly[a]).predict(Xv))
        s_all.append(r2_score(y[b], p))
        keep = y[b] <= 2000
        s_clip.append(r2_score(y[b][keep], p[keep]))
    print(f"{name:40s} R2 {np.mean(s_all):.4f} ± {np.std(s_all):.4f} | folds {np.round(s_all,3)} "
          f"| R2 w/o >$2000 rows {np.mean(s_clip):.4f}", flush=True)

run2("B: log price", False, False)
run2("E: log + flight target-encoding", True, False)
run2("F: log + target-enc + neighbours", True, True)
