#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fraud_app.py — Fraud tahmin arayüzü (yerel web uygulaması)

Notebook'ta seçilen modeli (model_sonuclari/ozet.json) ham CSV'den yeniden eğitir ve
tarayıcıda bir form olarak sunar: işlem bilgilerini gir -> "FRAUD / NORMAL" kararını gör.

Kullanım (CSV ile aynı klasörde):
    python fraud_app.py
    -> http://localhost:8100 açılır

Ek paket gerekmez: pandas, numpy, scikit-learn zaten kurulu. Sunucu için Python'un kendi http.server'ı kullanılır.
"""
import argparse
import ast
import csv
import json
import math
import os
import re
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

RS = 42
ID_RE = re.compile(r"(^|_)id($|_)")
TIME_NAMES = ("timestamp", "time", "date", "datetime", "trans_date_trans_time")

LABELS = {
    "transaction_amount": ("İşlem tutarı", "$"), "transaction_type": ("İşlem türü", ""), "timestamp": ("İşlem zamanı", ""),
    "merchant_category": ("Satıcı kategorisi", ""), "location": ("Konum", ""), "transaction_distance": ("İşlem mesafesi", "km"),
    "account_balance": ("Hesap bakiyesi", "$"), "card_type": ("Kart türü", ""), "card_age": ("Kart yaşı", "ay"),
    "avg_transaction_amount_7d": ("Son 7 gün ort. tutar", "$"), "daily_transaction_count": ("Günlük işlem sayısı", ""),
    "device_type": ("Cihaz", ""), "authentication_method": ("Doğrulama yöntemi", ""), "ip_address_flag": ("Şüpheli IP", ""),
    "previous_fraudulent_activity": ("Önceki fraud geçmişi", ""), "failed_transaction_count_7d": ("Son 7 gün başarısız işlem", ""),
    "risk_score": ("Risk skoru", ""), "is_weekend": ("Hafta sonu", ""),
}
GROUPS = [
    ("İşlem Bilgisi", "💳", ["transaction_amount", "transaction_type", "timestamp", "merchant_category", "location", "transaction_distance"]),
    ("Hesap ve Kart", "🏦", ["account_balance", "card_type", "card_age", "avg_transaction_amount_7d", "daily_transaction_count"]),
    ("Güvenlik ve Risk", "🛡️", ["device_type", "authentication_method", "ip_address_flag", "previous_fraudulent_activity",
                                  "failed_transaction_count_7d", "risk_score"]),
]


# --------------------------------------------------------------------------
# Özellik üretimi (EDA notebook'undaki satır bazlı adımların aynısı)
# --------------------------------------------------------------------------
def find_time_col(cols):
    low = {str(c).lower(): c for c in cols}
    return next((low[k] for k in TIME_NAMES if k in low), None)


def featurize(d):
    d = d.copy()
    low = {str(c).lower(): c for c in d.columns}
    tcol = find_time_col(d.columns)
    if tcol is not None:
        dt = pd.to_datetime(d[tcol], errors="coerce")
        d["saat"] = dt.dt.hour
        d["haftanin_gunu"] = dt.dt.dayofweek
        d["ay"] = dt.dt.month
        d["gece_mi"] = dt.dt.hour.between(0, 5).astype(int)
        d = d.drop(columns=[tcol])
    t, av, bal = low.get("transaction_amount"), low.get("avg_transaction_amount_7d"), low.get("account_balance")
    if t and av:
        d["tutar_ort_orani"] = d[t] / (d[av] + 1)
    if t and bal:
        d["tutar_bakiye_orani"] = d[t] / (d[bal].abs() + 1)
    fl, dy = low.get("failed_transaction_count_7d"), low.get("daily_transaction_count")
    if fl and dy:
        d["basarisiz_gunluk_orani"] = d[fl] / (d[dy] + 1)
    ds, ag = low.get("transaction_distance"), low.get("card_age")
    if ds and ag:
        d["uzaklik_kart_yasi_orani"] = d[ds] / (d[ag] + 1)
    return d


# --------------------------------------------------------------------------
# Model kurulumu
# --------------------------------------------------------------------------
def parse_param(v):
    if isinstance(v, str):
        if v == "None":
            return None
        try:
            return ast.literal_eval(v)
        except Exception:
            return v
    return v


def load_summary(path):
    if path and os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                j = json.load(f)
            j["parametreler"] = {k: parse_param(v) for k, v in j.get("parametreler", {}).items()}
            return j
        except Exception as e:
            print("uyarı: ozet.json okunamadı (%s), varsayılanlar kullanılacak" % e)
    return {}


def make_model(name, params, strategy, pos_weight):
    p = dict(params or {})
    if name == "HistGradientBoosting":
        m = HistGradientBoostingClassifier(random_state=RS, **p)
        if strategy == "class_weight":
            m.set_params(class_weight="balanced")
        return m, False
    if name == "LogisticRegression":
        p.setdefault("max_iter", 3000)
        m = LogisticRegression(**p)
        if strategy == "class_weight":
            m.set_params(class_weight="balanced")
        return m, True
    if name == "XGBoost":
        try:
            from xgboost import XGBClassifier
            p.setdefault("eval_metric", "logloss")
            m = XGBClassifier(tree_method="hist", random_state=RS, n_jobs=1, verbosity=0, **p)
            if strategy == "class_weight":
                m.set_params(scale_pos_weight=pos_weight)
            return m, False
        except Exception:
            print("uyarı: xgboost kullanılamıyor, RandomForest'a dönülüyor"); name = "RandomForest"; p = {}
    if name == "LightGBM":
        try:
            from lightgbm import LGBMClassifier
            m = LGBMClassifier(random_state=RS, n_jobs=1, verbose=-1, **p)
            if strategy == "class_weight":
                m.set_params(class_weight="balanced")
            return m, False
        except Exception:
            print("uyarı: lightgbm kullanılamıyor, RandomForest'a dönülüyor"); name = "RandomForest"; p = {}
    p.setdefault("n_estimators", 300)
    m = RandomForestClassifier(random_state=RS, n_jobs=1, **p)
    if strategy == "class_weight":
        m.set_params(class_weight="balanced")
    return m, False


def tree_rules(tree, names, min_share=0.01):
    """Derinlik-3 ağaçtan 'fraud' diyen yaprakların kurallarını okunur metne çevirir."""
    t = tree.tree_
    rules = []

    def walk(node, conds):
        if t.children_left[node] == -1:
            v = t.value[node][0]                       # yeni sklearn'de oran, eskide sayı: ikisinde de çalışsın
            purity = float(v[1] / max(v.sum(), 1e-12))
            n = int(t.n_node_samples[node])
            if purity > 0.5 and n / t.n_node_samples[0] >= min_share:
                rules.append({"if": conds, "n": n, "purity": purity})
            return
        f, thr = names[t.feature[node]], t.threshold[node]
        walk(t.children_left[node], conds + [(f, "<=", float(thr))])
        walk(t.children_right[node], conds + [(f, ">", float(thr))])

    walk(0, [])
    # Gereksiz tekrarlı koşulları sadeleştir ve metne çevir
    out = []
    for r in sorted(rules, key=lambda x: -x["n"]):
        best = {}
        for f, op, v in r["if"]:
            key = (f, op)
            if key not in best or (op == "<=" and v < best[key]) or (op == ">" and v > best[key]):
                best[key] = v
        txt = " VE ".join("%s %s %s" % (f, op, ("%.2f" % v).rstrip("0").rstrip(".")) for (f, op), v in best.items())
        out.append({"rule": txt, "n": r["n"], "purity": r["purity"]})
    return out


def build(args):
    df = pd.read_csv(args.csv)
    target = args.target
    if target not in df.columns:
        cand = [c for c in df.columns if str(c).lower() in ("fraud_label", "isfraud", "is_fraud", "fraud", "class", "label")]
        target = cand[0] if cand else df.columns[-1]
        print("Hedef sütun: %s" % target)
    y = df[target].astype(int)
    raw = df.drop(columns=[target])
    raw = raw.drop(columns=[c for c in raw.columns if ID_RE.search(str(c).lower())])
    tcol = find_time_col(raw.columns)

    Xtr, Xte, ytr, yte = train_test_split(raw, y, test_size=0.20, stratify=y, random_state=RS)

    fe_train = featurize(Xtr)
    cat_cols = [c for c in fe_train.columns if not pd.api.types.is_numeric_dtype(fe_train[c])]
    num_cols = [c for c in fe_train.columns if c not in cat_cols]

    ozet = load_summary(args.summary)
    name = ozet.get("model", "RandomForest")
    strategy = ozet.get("strateji", "class_weight")
    params = ozet.get("parametreler") or {"n_estimators": 300, "max_depth": 8, "min_samples_leaf": 20, "max_features": "sqrt"}
    threshold = float(ozet.get("esik", 0.5))
    pos_w = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    model, need_scale = make_model(name, params, strategy, pos_w)

    num_tf = StandardScaler() if need_scale else "passthrough"
    ct = ColumnTransformer([("num", num_tf, num_cols),
                            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)])
    pipe = Pipeline([("fe", FunctionTransformer(featurize, validate=False)), ("ct", ct), ("clf", model)])

    print("Model eğitiliyor: %s (%s) ..." % (name, strategy), flush=True)
    t0 = time.time()
    pipe.fit(Xtr, ytr)
    print("  eğitim bitti (%.1f sn)" % (time.time() - t0), flush=True)

    s_te = pipe.predict_proba(Xte)[:, 1]
    yh = (s_te >= threshold).astype(int)
    metrics = {"pr_auc": float(average_precision_score(yte, s_te)), "roc_auc": float(roc_auc_score(yte, s_te)),
               "precision": float(precision_score(yte, yh, zero_division=0)), "recall": float(recall_score(yte, yh)),
               "f1": float(f1_score(yte, yh)), "n_test": int(len(yte)), "fraud_rate": float(yte.mean())}
    s_tr = pipe.predict_proba(Xtr.sample(min(len(Xtr), 20000), random_state=RS))[:, 1]
    metrics["train_pr_auc"] = float(average_precision_score(ytr.loc[Xtr.sample(min(len(Xtr), 20000), random_state=RS).index], s_tr))

    # Basit kural testi (3 seviyeli ağaç): etiket basit bir kuraldan mı üretilmiş?
    ohe_tr = pd.get_dummies(fe_train, columns=cat_cols, dtype=float)
    ohe_te = pd.get_dummies(featurize(Xte), columns=cat_cols, dtype=float).reindex(columns=ohe_tr.columns, fill_value=0)
    dt = DecisionTreeClassifier(max_depth=3, min_samples_leaf=20, random_state=RS).fit(ohe_tr, ytr)
    dt_ap = float(average_precision_score(yte, dt.predict_proba(ohe_te)[:, 1]))
    rules = tree_rules(dt, list(ohe_tr.columns))
    rule_flag = dt_ap > 0.95

    # Form şeması (CSV'den otomatik)
    low = {str(c).lower(): c for c in raw.columns}
    fields, baseline = [], {}
    for c in raw.columns:
        lc = str(c).lower()
        lab, unit = LABELS.get(lc, (str(c).replace("_", " "), ""))
        f = {"name": c, "label": lab, "unit": unit}
        s = raw[c]
        if c == tcol:
            dtm = pd.to_datetime(s, errors="coerce").dropna()
            mid = dtm.sort_values().iloc[len(dtm) // 2]
            f.update(type="datetime", default=mid.strftime("%Y-%m-%dT%H:%M"), min=dtm.min().strftime("%Y-%m-%dT%H:%M"),
                     max=dtm.max().strftime("%Y-%m-%dT%H:%M"))
            baseline[c] = mid.strftime("%Y-%m-%d %H:%M:%S")
        elif lc == "is_weekend" and tcol:
            f["derived"] = True
            baseline[c] = float(s.median())
        elif not pd.api.types.is_numeric_dtype(s):
            vc = s.value_counts()
            f.update(type="select", options=[str(v) for v in vc.index], default=str(vc.index[0]))
            baseline[c] = vc.index[0]
        elif set(s.dropna().unique()) <= {0, 1}:
            f.update(type="toggle", default=int(s.mode().iloc[0]))
            baseline[c] = int(s.mode().iloc[0])
        else:
            lo, hi, med = float(s.min()), float(s.max()), float(s.median())
            isint = bool(np.all(np.mod(s.dropna(), 1) == 0))
            rng = max(hi - lo, 1e-9)
            step = 1 if isint else next((x for x in (0.0001, 0.001, 0.01, 0.1, 1, 10, 100) if x >= rng / 300), 1)
            f.update(type="number", min=lo, max=hi, step=step, isint=isint, default=int(med) if isint else round(med, 4))
            baseline[c] = f["default"]
        fields.append(f)

    grouped, used = [], set()
    for gname, icon, keys in GROUPS:
        fl = [f for k in keys for f in fields if str(f["name"]).lower() == k and not f.get("derived")]
        used.update(f["name"] for f in fl)
        if fl:
            grouped.append({"title": gname, "icon": icon, "fields": fl})
    rest = [f for f in fields if f["name"] not in used and not f.get("derived")]
    if rest:
        grouped.append({"title": "Diğer", "icon": "🧩", "fields": rest})

    # Model yarışı (varsa)
    race = []
    if args.lazy and os.path.exists(args.lazy):
        try:
            with open(args.lazy, encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            vals = [(r["Model"], float(r["PR_AUC (doğrulama)"]), float(r["PR_AUC (eğitim)"]), float(r["Fark (gap)"])) for r in rows]
            best = max(v[1] for v in vals if "Dummy" not in v[0])
            for m, va, tr, g in vals:
                if "Dummy" in m:
                    d = "taban"
                elif g > 0.05:
                    d = "overfit"
                elif va < best - 0.10:
                    d = "underfit"
                else:
                    d = "dengeli"
                race.append({"model": m, "val": va, "train": tr, "gap": g, "diag": d})
        except Exception as e:
            print("uyarı: lazy_yaris.csv okunamadı (%s)" % e)

    return {
        "pipe": pipe, "raw": raw, "Xte": Xte, "yte": yte, "baseline": baseline, "threshold": threshold, "tcol": tcol,
        "info": {
            "model": name, "strategy": strategy, "params": {k: str(v) for k, v in params.items()}, "threshold": threshold,
            "metrics": metrics, "rules": rules[:4], "rule_flag": rule_flag, "tree_ap": dt_ap,
            "n_features": int(len(num_cols) + len(cat_cols)), "race": race, "target": target, "csv": os.path.basename(args.csv),
            "fraud_rate": float(y.mean()),
        },
        "groups": grouped, "fields": {f["name"]: f for f in fields},
    }


# --------------------------------------------------------------------------
# Tahmin
# --------------------------------------------------------------------------
def to_frame(app, values):
    row = {}
    for c, f in app["fields"].items():
        v = values.get(c, None)
        if f.get("derived"):
            continue
        if v is None or v == "":
            v = app["baseline"][c]
        if f.get("type") == "datetime":
            v = str(v).replace("T", " ")
        elif f.get("type") in ("number", "toggle"):
            v = float(v)
        row[c] = v
    d = pd.DataFrame([row])
    tcol = app["tcol"]
    for c, f in app["fields"].items():
        if f.get("derived") and tcol:
            d[c] = int(pd.to_datetime(d[tcol]).dt.dayofweek.iloc[0] >= 5)
    return d[list(app["raw"].columns)]


def predict(app, values):
    d = to_frame(app, values)
    p = float(app["pipe"].predict_proba(d)[:, 1][0])
    contrib = []
    for c, f in app["fields"].items():
        if f.get("derived"):
            continue
        d2 = d.copy()
        b = app["baseline"][c]
        d2[c] = str(b) if f.get("type") == "datetime" else b
        if f.get("type") == "datetime" and app["tcol"]:
            for c2, f2 in app["fields"].items():
                if f2.get("derived"):
                    d2[c2] = int(pd.to_datetime(d2[app["tcol"]]).dt.dayofweek.iloc[0] >= 5)
        p2 = float(app["pipe"].predict_proba(d2)[:, 1][0])
        contrib.append({"field": c, "delta": p - p2})
    contrib.sort(key=lambda x: -abs(x["delta"]))
    return {"prob": p, "fraud": bool(p >= app["threshold"]), "threshold": app["threshold"], "contrib": contrib[:6]}


def sample(app, kind):
    y, X = app["yte"], app["Xte"]
    idx = y.index
    if kind == "fraud":
        idx = y[y == 1].index
    elif kind == "normal":
        idx = y[y == 0].index
    i = np.random.choice(idx)
    row = X.loc[i]
    vals = {}
    for c, f in app["fields"].items():
        v = row[c]
        if f.get("type") == "datetime":
            v = pd.to_datetime(v).strftime("%Y-%m-%dT%H:%M")
        elif f.get("type") in ("number", "toggle"):
            v = float(v) if not f.get("isint") and f.get("type") == "number" else int(v)
        else:
            v = str(v)
        vals[c] = v
    return {"values": vals, "actual": int(y.loc[i])}


# --------------------------------------------------------------------------
# Web arayüzü
# --------------------------------------------------------------------------
PAGE = r'''<!DOCTYPE html>
<html lang="tr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>FraudShield · Canlı Fraud Tahmini</title>
<style>
:root{--bg:#070b1a;--glass:rgba(255,255,255,.06);--line:rgba(255,255,255,.13);--gold:#f5c451;--gold2:#ffe29a;--teal:#2dd4bf;--red:#ff4d6d;--txt:#e8ecf8;--mut:#9aa6c7;--ok:#34d399}
*{box-sizing:border-box;margin:0;padding:0}
html,body{min-height:100%;background:var(--bg);color:var(--txt);font-family:"Segoe UI",Inter,system-ui,-apple-system,Arial,sans-serif}
body:before{content:"";position:fixed;inset:0;z-index:0;pointer-events:none;background:radial-gradient(900px 500px at 10% 0%,rgba(245,196,81,.12),transparent 60%),radial-gradient(800px 600px at 95% 100%,rgba(45,212,191,.10),transparent 60%),radial-gradient(700px 500px at 85% 0%,rgba(124,92,255,.13),transparent 60%)}
#rain{position:fixed;inset:0;z-index:0;opacity:.5}
header{position:sticky;top:0;z-index:30;display:flex;align-items:center;justify-content:space-between;padding:14px 28px;backdrop-filter:blur(12px);background:linear-gradient(180deg,rgba(7,11,26,.92),rgba(7,11,26,.4));border-bottom:1px solid var(--line)}
.brand{display:flex;align-items:center;gap:12px}.minicard{width:36px;height:24px;border-radius:5px;background:linear-gradient(135deg,var(--gold),#c98f1c);position:relative;box-shadow:0 4px 14px rgba(245,196,81,.35)}
.minicard:after{content:"";position:absolute;left:5px;top:6px;width:9px;height:7px;border-radius:2px;background:rgba(7,11,26,.55)}
.brand b{letter-spacing:.18em;font-size:14px}.brand small{display:block;color:var(--mut);font-size:10.5px;letter-spacing:.08em;margin-top:2px}
.pill{font-size:12px;color:var(--mut);display:flex;gap:10px;align-items:center;flex-wrap:wrap;justify-content:flex-end}
.pill span{background:var(--glass);border:1px solid var(--line);padding:5px 11px;border-radius:999px}.pill b{color:var(--gold2)}
main{position:relative;z-index:5;display:grid;grid-template-columns:minmax(0,1.25fr) minmax(360px,.9fr);gap:22px;padding:22px 28px 60px;max-width:1500px;margin:auto}
@media(max-width:1000px){main{grid-template-columns:1fr}}
.panel{background:var(--glass);border:1px solid var(--line);border-radius:20px;padding:18px 20px;backdrop-filter:blur(10px);margin-bottom:16px}
.panel h3{font-size:13px;letter-spacing:.16em;color:var(--gold);margin-bottom:14px;display:flex;align-items:center;gap:10px}
.toolbar{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px;align-items:center}
.btn{background:var(--glass);border:1px solid var(--line);color:var(--txt);padding:9px 14px;border-radius:12px;cursor:pointer;font-size:13px;transition:.25s}
.btn:hover{border-color:var(--gold);color:var(--gold2);transform:translateY(-2px)}
.btn.gold{background:linear-gradient(135deg,var(--gold2),var(--gold));color:#1a1300;border-color:transparent;font-weight:700}
.btn.gold:hover{color:#1a1300;box-shadow:0 10px 28px rgba(245,196,81,.4)}
.auto{margin-left:auto;font-size:12.5px;color:var(--mut);display:flex;align-items:center;gap:8px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:14px 16px}
.f label{display:flex;justify-content:space-between;font-size:12px;color:var(--mut);margin-bottom:6px;letter-spacing:.03em}
.f label em{font-style:normal;color:var(--gold2);font-variant-numeric:tabular-nums}
.f select,.f input[type=number],.f input[type=datetime-local]{width:100%;background:rgba(7,11,26,.7);border:1px solid var(--line);color:var(--txt);padding:9px 11px;border-radius:10px;font-size:14px;outline:none;color-scheme:dark}
.f select:focus,.f input:focus{border-color:var(--gold);box-shadow:0 0 0 3px rgba(245,196,81,.15)}
.f input[type=range]{width:100%;accent-color:var(--gold);margin-top:7px}
.sw{display:flex;align-items:center;gap:10px;background:rgba(7,11,26,.7);border:1px solid var(--line);padding:8px 11px;border-radius:10px;cursor:pointer}
.sw input{display:none}.sw i{width:38px;height:21px;border-radius:99px;background:rgba(255,255,255,.14);position:relative;transition:.25s;flex:none}
.sw i:after{content:"";position:absolute;left:3px;top:3px;width:15px;height:15px;border-radius:50%;background:#fff;transition:.25s}
.sw input:checked+i{background:var(--red)}.sw input:checked+i:after{left:20px}.sw span{font-size:13px;color:var(--txt)}
/* sonuç paneli */
.side{position:sticky;top:84px;align-self:start}
.verdict{position:relative;border-radius:22px;padding:22px;overflow:hidden;border:1px solid var(--line);background:linear-gradient(135deg,#16245a,#0b1233 55%,#2c1d5e);transition:box-shadow .5s,border-color .5s}
.verdict.ok{box-shadow:0 0 70px rgba(52,211,153,.22);border-color:rgba(52,211,153,.5)}
.verdict.bad{box-shadow:0 0 80px rgba(255,77,109,.35);border-color:rgba(255,77,109,.6);animation:shake .5s}
@keyframes shake{25%{transform:translateX(-6px)}75%{transform:translateX(6px)}}
.verdict .holo{position:absolute;inset:0;background:linear-gradient(115deg,transparent 20%,rgba(255,255,255,.1) 35%,rgba(45,212,191,.12) 45%,rgba(245,196,81,.14) 55%,transparent 70%);background-size:250% 250%;animation:holo 5s linear infinite;mix-blend-mode:screen;pointer-events:none}
@keyframes holo{from{background-position:130% 0}to{background-position:-130% 0}}
.beam{position:absolute;left:0;right:0;height:70px;top:-80px;background:linear-gradient(180deg,transparent,rgba(45,212,191,.5),transparent);pointer-events:none}
.verdict.scan .beam{animation:scan 1s ease-in-out}@keyframes scan{to{top:110%}}
.vtop{display:flex;justify-content:space-between;align-items:center;position:relative;z-index:2}
.vtop b{letter-spacing:.2em;font-size:12px}
.chip{width:46px;height:34px;border-radius:7px;background:linear-gradient(135deg,#ffe29a,#e0a82e 55%,#b9821a);position:relative;z-index:2;margin:14px 0 6px}
.chip:before,.chip:after{content:"";position:absolute;left:0;right:0;height:1px;background:rgba(0,0,0,.35)}.chip:before{top:33%}.chip:after{top:66%}
.stamp{position:relative;z-index:2;font-weight:900;font-size:clamp(26px,3.4vw,40px);letter-spacing:.06em;margin:6px 0 2px;transition:.4s}
.stamp.ok{color:var(--ok)}.stamp.bad{color:var(--red)}.stamp.wait{color:var(--mut)}
.sub{position:relative;z-index:2;color:var(--mut);font-size:13px;min-height:20px}
.gauge{position:relative;z-index:2;margin:10px auto 0;width:100%;max-width:330px}
.gauge svg{width:100%;display:block}
.gauge .pct{position:absolute;left:0;right:0;bottom:2px;text-align:center}
.gauge .pct b{font-size:38px;font-weight:900;font-variant-numeric:tabular-nums}.gauge .pct small{display:block;color:var(--mut);font-size:11px;letter-spacing:.12em}
#arc{transition:stroke-dashoffset 1s cubic-bezier(.2,.8,.2,1),stroke .5s}
.facts .row{display:grid;grid-template-columns:minmax(120px,1.1fr) 2fr 56px;gap:10px;align-items:center;font-size:12.5px;margin:9px 0}
.facts .name{color:var(--txt);line-height:1.25}.facts .name small{display:block;color:var(--mut);font-size:11px}
.facts .bar{height:9px;border-radius:9px;background:rgba(255,255,255,.08);position:relative;overflow:hidden}
.facts .bar i{position:absolute;top:0;bottom:0;border-radius:9px;transition:width .8s cubic-bezier(.2,.8,.2,1)}
.facts .bar i.up{background:linear-gradient(90deg,#ff8fa3,var(--red));left:50%}.facts .bar i.down{background:linear-gradient(270deg,#7ae7d6,var(--teal));right:50%}
.facts .bar:after{content:"";position:absolute;left:50%;top:-2px;bottom:-2px;width:1px;background:var(--mut)}
.facts .v{text-align:right;font-variant-numeric:tabular-nums;color:var(--mut)}
.actual{margin-top:12px;padding:10px 14px;border-radius:12px;font-size:13px;display:none}
.actual.on{display:block}.actual.good{background:rgba(52,211,153,.12);border:1px solid rgba(52,211,153,.4);color:var(--ok)}.actual.badx{background:rgba(255,77,109,.12);border:1px solid rgba(255,77,109,.4);color:var(--red)}
.note{margin-bottom:16px;padding:13px 16px;border-radius:14px;background:rgba(245,196,81,.1);border:1px solid rgba(245,196,81,.5);color:var(--gold2);font-size:13px;line-height:1.55}
.note code{background:rgba(0,0,0,.3);padding:1px 6px;border-radius:6px;color:#fff}
.note ul{margin:6px 0 0 18px}
.tabs{display:flex;gap:6px;margin-bottom:14px;flex-wrap:wrap}.tabs button{background:none;border:1px solid var(--line);color:var(--mut);padding:7px 14px;border-radius:999px;cursor:pointer;font-size:12.5px}
.tabs button.on{background:linear-gradient(135deg,var(--gold2),var(--gold));color:#1a1300;font-weight:700;border-color:transparent}
.kv{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px;margin-bottom:12px}
.kv div{background:rgba(7,11,26,.6);border:1px solid var(--line);border-radius:12px;padding:10px 12px}.kv b{display:block;font-size:19px;color:var(--gold2);font-variant-numeric:tabular-nums}.kv span{font-size:11px;color:var(--mut);letter-spacing:.06em}
.race .r{display:grid;grid-template-columns:150px 1fr 82px;gap:10px;align-items:center;font-size:12.5px;margin:6px 0}
.race .bar{height:12px;border-radius:8px;background:rgba(255,255,255,.07);position:relative;overflow:hidden}.race .bar i{position:absolute;left:0;top:0;bottom:0;border-radius:8px}
.race .tag{font-size:11px;text-align:center;padding:2px 6px;border-radius:99px}
.t-dengeli{background:rgba(52,211,153,.18);color:var(--ok)}.t-overfit{background:rgba(255,77,109,.18);color:var(--red)}.t-underfit{background:rgba(245,196,81,.18);color:var(--gold2)}.t-taban{background:rgba(255,255,255,.1);color:var(--mut)}
.hide{display:none}.small{font-size:12px;color:var(--mut);line-height:1.6}
</style></head>
<body>
<canvas id="rain"></canvas>
<header><div class="brand"><div class="minicard"></div><div><b>FRAUDSHIELD</b><small>CANLI İŞLEM RİSK MOTORU</small></div></div>
<div class="pill" id="pill"></div></header>
<main>
 <section>
  <div id="note"></div>
  <div class="toolbar">
    <button class="btn" data-s="any">🎲 Veriden rastgele işlem</button>
    <button class="btn" data-s="fraud">🚨 Gerçek bir fraud örneği</button>
    <button class="btn" data-s="normal">✅ Gerçek bir normal örnek</button>
    <button class="btn" id="reset">↺ Ortalama işlem</button>
    <label class="auto"><input type="checkbox" id="auto" checked> Değiştikçe otomatik tahmin et</label>
  </div>
  <div id="form"></div>
  <button class="btn gold" id="go" style="width:100%;padding:14px;font-size:15px">🔍 Tahmin Et</button>
 </section>
 <aside class="side">
  <div class="verdict wait" id="verdict"><div class="holo"></div><div class="beam"></div>
   <div class="vtop"><b>FRAUDSHIELD BANK</b><span style="font-size:12px;color:var(--mut)" id="tx">TXN · —</span></div>
   <div class="chip"></div>
   <div class="stamp wait" id="stamp">HAZIR</div>
   <div class="sub" id="sub">Soldaki formu doldur ve "Tahmin Et"e bas.</div>
   <div class="gauge"><svg viewBox="0 0 200 118"><path d="M20 105 A80 80 0 0 1 180 105" fill="none" stroke="rgba(255,255,255,.1)" stroke-width="16" stroke-linecap="round"/>
     <path id="arc" d="M20 105 A80 80 0 0 1 180 105" fill="none" stroke="#2dd4bf" stroke-width="16" stroke-linecap="round" stroke-dasharray="251.3" stroke-dashoffset="251.3"/>
     <line id="thr" x1="0" y1="0" x2="0" y2="0" stroke="#fff" stroke-width="2.5" stroke-linecap="round" opacity=".9"/></svg>
     <div class="pct"><b id="pct">—</b><small>FRAUD OLASILIĞI</small></div></div>
  </div>
  <div class="actual" id="actual"></div>
  <div class="panel facts" style="margin-top:16px"><h3>🧭 KARARA EN ÇOK ETKİ EDENLER</h3><div id="facts"><div class="small">Tahmin yapınca burada görünür. Kırmızı çubuk fraud olasılığını artıran, yeşil çubuk azaltan alandır (alanı ortalama değerine çekince olasılığın ne kadar değiştiği).</div></div></div>
  <div class="panel"><div class="tabs" id="tabs"><button class="on" data-t="m">Model</button><button data-t="r">Model yarışı</button></div>
   <div id="tm"></div><div id="tr" class="hide"></div></div>
 </aside>
</main>
<script id="schema" type="application/json">__SCHEMA__</script>
<script>
(function(){
'use strict';
var S=JSON.parse(document.getElementById('schema').textContent),I=S.info;
var $=function(s,r){return (r||document).querySelector(s)},$$=function(s,r){return [].slice.call((r||document).querySelectorAll(s))};
var esc=function(s){return String(s).replace(/[&<>"']/g,function(c){return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]})};
var pc=function(x,d){return (x*100).toFixed(d==null?1:d)+'%'};
var fields={};S.groups.forEach(function(g){g.fields.forEach(function(f){fields[f.name]=f})});

/* ---- form ---- */
var form=$('#form');
S.groups.forEach(function(g){
  var p=document.createElement('div');p.className='panel';
  var h='<h3>'+g.icon+' '+esc(g.title.toUpperCase())+'</h3><div class="grid">';
  g.fields.forEach(function(f){h+=fieldHtml(f)});
  p.innerHTML=h+'</div>';form.appendChild(p);
});
function fieldHtml(f){
  var u=f.unit?' <small>'+esc(f.unit)+'</small>':'',n=esc(f.name),lab=esc(f.label);
  if(f.type==='select')return '<div class="f"><label>'+lab+'</label><select data-n="'+n+'">'+f.options.map(function(o){return '<option'+(o===f.default?' selected':'')+'>'+esc(o)+'</option>'}).join('')+'</select></div>';
  if(f.type==='toggle')return '<div class="f"><label>'+lab+'</label><label class="sw"><input type="checkbox" data-n="'+n+'"'+(f.default?' checked':'')+'><i></i><span>'+(f.default?'Evet':'Hayır')+'</span></label></div>';
  if(f.type==='datetime')return '<div class="f"><label>'+lab+'</label><input type="datetime-local" data-n="'+n+'" value="'+f.default+'" min="'+f.min+'" max="'+f.max+'"></div>';
  return '<div class="f"><label>'+lab+u+'<em data-v="'+n+'">'+f.default+'</em></label><input type="range" data-r="'+n+'" min="'+f.min+'" max="'+f.max+'" step="'+f.step+'" value="'+f.default+'"><input type="number" data-n="'+n+'" min="'+f.min+'" max="'+f.max+'" step="'+f.step+'" value="'+f.default+'" style="margin-top:7px"></div>';
}
function setVal(n,v){
  var f=fields[n];if(!f)return;
  var el=$('[data-n="'+n+'"]');if(!el)return;
  if(f.type==='toggle'){el.checked=!!Number(v);var sp=el.parentNode.querySelector('span');if(sp)sp.textContent=el.checked?'Evet':'Hayır'}
  else if(f.type==='number'){el.value=v;var r=$('[data-r="'+n+'"]');if(r)r.value=v;var e=$('[data-v="'+n+'"]');if(e)e.textContent=v}
  else el.value=v;
}
function getVals(){
  var o={};$$('[data-n]').forEach(function(el){var n=el.dataset.n,f=fields[n];
    o[n]=f.type==='toggle'?(el.checked?1:0):el.value});return o;
}
form.addEventListener('input',function(e){
  var t=e.target;
  if(t.dataset.r){var n=t.dataset.r;$('[data-n="'+n+'"]').value=t.value;$('[data-v="'+n+'"]').textContent=t.value}
  else if(t.dataset.n&&fields[t.dataset.n].type==='number'){var r=$('[data-r="'+t.dataset.n+'"]');if(r)r.value=t.value;var em=$('[data-v="'+t.dataset.n+'"]');if(em)em.textContent=t.value}
  else if(t.type==='checkbox'){var sp=t.parentNode.querySelector('span');if(sp)sp.textContent=t.checked?'Evet':'Hayır'}
  hideActual();schedule();
});
var timer=null;function schedule(){if(!$('#auto').checked)return;clearTimeout(timer);timer=setTimeout(run,350)}

/* ---- tahmin ---- */
var ARC=251.3,seq=0;
function setThr(t){var a=Math.PI*(1-t),x1=100+68*Math.cos(a),y1=105-68*Math.sin(a),x2=100+92*Math.cos(a),y2=105-92*Math.sin(a);var l=$('#thr');l.setAttribute('x1',x1);l.setAttribute('y1',y1);l.setAttribute('x2',x2);l.setAttribute('y2',y2)}
setThr(I.threshold);
function run(){
  var my=++seq,v=$('#verdict');v.classList.remove('scan');void v.offsetWidth;v.classList.add('scan');
  fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(getVals())})
   .then(function(r){return r.json()}).then(function(r){if(my===seq)show(r)}).catch(function(){$('#sub').textContent='Sunucuya ulaşılamadı. fraud_app.py çalışıyor mu?'});
}
function show(r){
  var v=$('#verdict'),st=$('#stamp'),bad=r.fraud;
  v.className='verdict '+(bad?'bad':'ok')+' scan';st.className='stamp '+(bad?'bad':'ok');
  st.textContent=bad?'FRAUD ⚠':'NORMAL ✓';
  $('#sub').textContent=bad?'İşlem engellenmeli: olasılık eşiğin (%'+(r.threshold*100).toFixed(0)+') üstünde':'İşlem onaylanabilir: olasılık eşiğin (%'+(r.threshold*100).toFixed(0)+') altında';
  $('#tx').textContent='TXN · '+Math.floor(10000+Math.random()*89999);
  var arc=$('#arc');arc.style.strokeDashoffset=ARC*(1-r.prob);arc.style.stroke=r.prob<r.threshold*0.7?'#2dd4bf':r.prob<r.threshold?'#f5c451':'#ff4d6d';
  count($('#pct'),r.prob*100);
  var vals=getVals(),h='',mx=Math.max.apply(null,r.contrib.map(function(c){return Math.abs(c.delta)}).concat([0.05]));
  r.contrib.forEach(function(c){
    var f=fields[c.field]||{label:c.field},w=Math.min(50,Math.abs(c.delta)/mx*50),up=c.delta>0;
    var val=vals[c.field];if(f.type==='toggle')val=val?'Evet':'Hayır';
    h+='<div class="row"><div class="name">'+esc(f.label)+'<small>'+esc(val)+'</small></div><div class="bar"><i class="'+(up?'up':'down')+'" style="width:'+w+'%"></i></div><div class="v">'+(c.delta>0?'+':'')+(c.delta*100).toFixed(1)+' p</div></div>';
  });
  $('#facts').innerHTML=h||'<div class="small">Belirgin bir etken yok.</div>';
  if(window.__actual!=null){var a=window.__actual,good=(a===1)===bad;var el=$('#actual');el.className='actual on '+(good?'good':'badx');
    el.textContent='Veri setindeki gerçek etiket: '+(a?'FRAUD':'NORMAL')+' → model '+(good?'doğru bildi ✓':'yanıldı ✗');}
}
function count(el,to){var t0=null,from=parseFloat(el.dataset.v||0);el.dataset.v=to;
  function st(ts){if(!t0)t0=ts;var p=Math.min(1,(ts-t0)/700),e=1-Math.pow(1-p,3);el.textContent=(from+(to-from)*e).toFixed(1)+'%';if(p<1)requestAnimationFrame(st)}requestAnimationFrame(st)}
function hideActual(){window.__actual=null;$('#actual').className='actual'}
$('#go').onclick=run;

/* ---- örnekler ---- */
$$('[data-s]').forEach(function(b){b.onclick=function(){
  fetch('/sample?kind='+b.dataset.s).then(function(r){return r.json()}).then(function(s){
    Object.keys(s.values).forEach(function(n){setVal(n,s.values[n])});window.__actual=s.actual;run()})}});
$('#reset').onclick=function(){Object.keys(fields).forEach(function(n){setVal(n,fields[n].default)});hideActual();run()};

/* ---- üst bilgi ---- */
var m=I.metrics;
$('#pill').innerHTML='<span>Model: <b>'+esc(I.model)+'</b></span><span>Test PR-AUC: <b>'+m.pr_auc.toFixed(3)+'</b></span><span>Recall: <b>'+pc(m.recall,1)+'</b></span><span>Eşik: <b>'+I.threshold.toFixed(2)+'</b></span>';
if(I.rule_flag){
  var rs=I.rules.slice(0,3).map(function(r){return '<li><code>'+esc(r.rule)+'</code> → fraud <small>('+r.n+' eğitim satırı, %'+(r.purity*100).toFixed(1)+' saf)</small></li>'}).join('');
  $('#note').innerHTML='<div class="note"><b>⚠️ Bu veri setinde etiket basit bir kuraldan üretilmiş görünüyor.</b> 3 seviyeli minicik bir karar ağacı bile test setinde PR-AUC=<b>'+I.tree_ap.toFixed(3)+'</b> alıyor. Model bu kuralı öğrendi; diğer alanlar (tutar, konum, cihaz…) karara neredeyse hiç etki etmiyor. Yani yüksek skor, modelin zekâsından çok verinin yapısından geliyor:<ul>'+rs+'</ul></div>';
}
/* ---- model sekmeleri ---- */
var ph=Object.keys(I.params).map(function(k){return '<code style="background:rgba(0,0,0,.3);padding:1px 6px;border-radius:6px">'+esc(k)+'='+esc(I.params[k])+'</code>'}).join(' ');
$('#tm').innerHTML='<div class="kv"><div><b>'+m.pr_auc.toFixed(3)+'</b><span>TEST PR-AUC</span></div><div><b>'+pc(m.recall,1)+'</b><span>RECALL</span></div><div><b>'+pc(m.precision,1)+'</b><span>PRECISION</span></div><div><b>'+m.f1.toFixed(3)+'</b><span>F1</span></div></div>'+
 '<div class="small">Algoritma: <b>'+esc(I.model)+'</b> · dengesizlik: <b>'+esc(I.strategy)+'</b> · eşik: <b>'+I.threshold.toFixed(3)+'</b><br>Parametreler: '+ph+'<br>Eğitim PR-AUC: <b>'+m.train_pr_auc.toFixed(3)+'</b> · test PR-AUC: <b>'+m.pr_auc.toFixed(3)+'</b> (fark küçükse overfit yok)<br>Veri: '+esc(I.csv)+' · fraud oranı %'+(I.fraud_rate*100).toFixed(1)+' · test seti '+m.n_test.toLocaleString('tr-TR')+' işlem<br><br>Not: model, ham CSV’den notebook’ta seçilen ayarlarla yeniden eğitilir (bölme: %80 eğitim / %20 test, aynı tohum).</div>';
if(I.race.length){
  var rh='<div class="small" style="margin-bottom:8px">Doğrulama PR-AUC (çubuk) ve eğitim−doğrulama farkı (gap). <span class="tag t-dengeli">dengeli</span> <span class="tag t-overfit">overfit</span> <span class="tag t-underfit">underfit</span></div><div class="race">';
  I.race.forEach(function(r){var col=r.diag==='overfit'?'#ff4d6d':r.diag==='underfit'?'#f5c451':r.diag==='taban'?'#8d99ae':'#2dd4bf';
    rh+='<div class="r"><div>'+esc(r.model)+'</div><div class="bar"><i style="width:'+(r.val*100)+'%;background:'+col+'"></i></div><div><span class="tag t-'+r.diag+'">'+r.val.toFixed(3)+' · Δ'+r.gap.toFixed(3)+'</span></div></div>'});
  $('#tr').innerHTML=rh+'</div>';
}else{$('#tr').innerHTML='<div class="small">model_sonuclari/lazy_yaris.csv bulunamadı; modelleme notebook’unu çalıştırınca burada model yarışı görünür.</div>'}
$$('#tabs button').forEach(function(b){b.onclick=function(){$$('#tabs button').forEach(function(x){x.classList.toggle('on',x===b)});$('#tm').classList.toggle('hide',b.dataset.t!=='m');$('#tr').classList.toggle('hide',b.dataset.t!=='r')}});

/* ---- arka plan ---- */
(function(){try{var c=$('#rain'),x=c.getContext('2d');if(!x)return;var w,h,cols,d,G='01$€₺%¥£#'.split('');
 function sz(){w=c.width=innerWidth;h=c.height=innerHeight;cols=Math.floor(w/28);d=[];for(var i=0;i<cols;i++)d.push(Math.random()*h/28)}sz();addEventListener('resize',sz);
 if(matchMedia&&matchMedia('(prefers-reduced-motion: reduce)').matches)return;
 setInterval(function(){x.fillStyle='rgba(7,11,26,.14)';x.fillRect(0,0,w,h);x.font='16px monospace';
  for(var i=0;i<cols;i++){x.fillStyle=Math.random()>.97?'rgba(245,196,81,.5)':'rgba(45,212,191,.14)';x.fillText(G[Math.floor(Math.random()*G.length)],i*28,d[i]*28);if(d[i]*28>h&&Math.random()>.975)d[i]=0;d[i]+=.5}},60)}catch(e){}})();
run();
window.__app={run:run,show:show};
})();
</script></body></html>
'''


class Handler(BaseHTTPRequestHandler):
    app = None
    page = b""

    def log_message(self, *a):
        pass

    def _send(self, body, ctype="application/json; charset=utf-8", code=200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path, _, qs = self.path.partition("?")
        if path in ("/", "/index.html"):
            self._send(self.page, "text/html; charset=utf-8")
        elif path == "/sample":
            kind = "any"
            for part in qs.split("&"):
                if part.startswith("kind="):
                    kind = part[5:]
            self._send(json.dumps(sample(self.app, kind), ensure_ascii=False).encode("utf-8"))
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path != "/predict":
            self.send_error(404); return
        try:
            n = int(self.headers.get("Content-Length", 0))
            values = json.loads(self.rfile.read(n).decode("utf-8") or "{}")
            self._send(json.dumps(predict(self.app, values), ensure_ascii=False).encode("utf-8"))
        except Exception as e:
            self._send(json.dumps({"error": str(e)}).encode("utf-8"), code=400)


def main():
    ap = argparse.ArgumentParser(description="Fraud tahmin arayüzü")
    ap.add_argument("--csv", default="synthetic_fraud_dataset.csv")
    ap.add_argument("--target", default="Fraud_Label")
    ap.add_argument("--summary", default=os.path.join("model_sonuclari", "ozet.json"))
    ap.add_argument("--lazy", default=os.path.join("model_sonuclari", "lazy_yaris.csv"))
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    if not os.path.exists(a.csv):
        sys.exit("CSV bulunamadı: %s  (fraud_app.py'yi CSV ile aynı klasörde çalıştır veya --csv ile yolu ver)" % a.csv)

    app = build(a)
    schema = {"groups": app["groups"], "info": app["info"]}
    payload = json.dumps(schema, ensure_ascii=False).replace("</", "<\\/")
    Handler.app = app
    Handler.page = PAGE.replace("__SCHEMA__", payload).encode("utf-8")

    server = None
    for port in range(a.port, a.port + 10):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
            break
        except OSError:
            continue
    if server is None:
        sys.exit("Boş port bulunamadı. --port ile başka bir port dene.")
    url = "http://localhost:%d" % server.server_address[1]
    m = app["info"]["metrics"]
    print("\nHazır -> %s" % url)
    print("Model: %s | test PR-AUC %.4f | recall %.4f | precision %.4f | eşik %.3f" %
          (app["info"]["model"], m["pr_auc"], m["recall"], m["precision"], app["info"]["threshold"]))
    print("Durdurmak için Ctrl+C\n", flush=True)
    if not a.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nKapatıldı.")


if __name__ == "__main__":
    main()
