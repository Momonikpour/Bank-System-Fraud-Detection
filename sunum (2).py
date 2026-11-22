#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sunum.py  —  Jupyter notebook'taki GRAFİKLERİ (kod olmadan) animasyonlu bir
"kredi kartı / banka" temalı HTML sunuma çevirir ve notebook her kaydedildiğinde
HTML'i otomatik günceller.

Kullanım (notebook ile aynı klasörde):
    python sunum.py                      -> sunucuyu başlatır, tarayıcıda http://localhost:8000 açar
    python sunum.py --nb baska.ipynb     -> farklı notebook
    python sunum.py --once               -> sadece sunum.html üretir (sunucu yok)

Ek paket gerekmez, sadece Python standart kütüphanesi kullanılır.
"""
import argparse
import html as htmllib
import json
import os
import re
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# --------------------------------------------------------------------------
# 1) Küçük yardımcılar: emoji temizleme ve mini markdown -> HTML dönüştürücü
# --------------------------------------------------------------------------
EMOJI_RE = re.compile("[\U0001F000-\U0001FAFF\u2300-\u23FF\u2600-\u27BF\u2B00-\u2BFF\uFE0F\u200d\u2194-\u21AA]+")


def strip_emoji(s):
    return EMOJI_RE.sub("", s).strip()


def md_inline(s):
    s = htmllib.escape(s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    return s


def md_plain(s):
    s = re.sub(r"`([^`]+)`", r"\1", s)
    s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)
    return strip_emoji(s).strip()


def md_table(rows):
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    cells = [r for r in cells if not all(re.fullmatch(r":?-{2,}:?", c) for c in r if c)]
    if not cells:
        return ""
    out = ["<table class='mdtable'><thead><tr>"]
    out += ["<th>%s</th>" % md_inline(c) for c in cells[0]]
    out.append("</tr></thead><tbody>")
    for r in cells[1:]:
        out.append("<tr>" + "".join("<td>%s</td>" % md_inline(c) for c in r) + "</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def md_to_html(text):
    lines = text.split("\n")
    out, para, i = [], [], 0

    def flush():
        if para:
            out.append("<p>%s</p>" % md_inline(" ".join(para)))
            para.clear()

    while i < len(lines):
        s = lines[i].strip()
        if not s:
            flush(); i += 1; continue
        if s.startswith("#"):
            flush(); i += 1; continue
        if s.startswith("|"):
            flush(); rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(lines[i].strip()); i += 1
            out.append(md_table(rows)); continue
        if re.match(r"^[-*]\s+", s) or re.match(r"^\d+\.\s+", s):
            flush()
            ordered = bool(re.match(r"^\d+\.\s+", s))
            items = []
            while i < len(lines):
                t = lines[i].strip()
                if re.match(r"^[-*]\s+", t) or re.match(r"^\d+\.\s+", t):
                    items.append(re.sub(r"^([-*]|\d+\.)\s+", "", t))
                elif t and lines[i][:1] in (" ", "\t") and items:
                    items[-1] += " " + t
                else:
                    break
                i += 1
            tag = "ol" if ordered else "ul"
            out.append("<%s>%s</%s>" % (tag, "".join("<li>%s</li>" % md_inline(x) for x in items), tag))
            continue
        if s.startswith(">"):
            flush(); q = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                q.append(lines[i].strip().lstrip(">").strip()); i += 1
            out.append("<blockquote>%s</blockquote>" % md_inline(" ".join(x for x in q if x)))
            continue
        para.append(s); i += 1
    flush()
    return "".join(out)


def first_sentence(text, limit=170):
    """Başlığın altındaki ilk paragraftan kısa bir alt başlık çıkarır."""
    for raw in text.split("\n"):
        s = raw.strip()
        if not s or s.startswith(("#", "|", ">", "---")):
            continue
        s = re.sub(r"^([-*]|\d+\.)\s+", "", s)
        s = md_plain(s)
        if len(s) < 8:
            continue
        if len(s) > limit:
            s = s[:limit].rsplit(" ", 1)[0] + "…"
        return s
    return ""


def clean_table_html(h):
    h = re.sub(r"<style.*?</style>", "", h, flags=re.S)
    h = re.sub(r'\s(border|style)="[^"]*"', "", h)
    return h


# --------------------------------------------------------------------------
# 2) Notebook'u oku -> slayt verisi
# --------------------------------------------------------------------------
def join(src):
    return "".join(src) if isinstance(src, list) else str(src)


def read_notebook(path):
    for _ in range(6):                       # Jupyter dosyayı yazarken yarım okumayalım
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            time.sleep(0.4)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def parse_kpis(text):
    k = {}
    m = re.search(r"Satır sayısı:\s*([\d,]+)\s*\|\s*Sütun sayısı:\s*(\d+)", text)
    if m:
        k["rows"] = int(m.group(1).replace(",", "")); k["cols"] = int(m.group(2))
    m = re.search(r"Normal \(0\):\s*([\d,]+)\s*\(%([\d.]+)\)", text)
    if m:
        k["normal"] = int(m.group(1).replace(",", ""))
    m = re.search(r"Fraud\s+\(1\):\s*([\d,]+)\s*\(%([\d.]+)\)", text)
    if m:
        k["fraud"] = int(m.group(1).replace(",", "")); k["fraud_pct"] = float(m.group(2))
    m = re.search(r"Seçilen (\d+) özellik", text)
    if m:
        k["selected"] = int(m.group(1))
    m = re.search(r"Seçilen hedef sütun: '([^']+)'|Elle verilen hedef sütun: '([^']+)'", text)
    if m:
        k["target"] = m.group(1) or m.group(2)
    m = re.search(r"Kullanılan dosya:\s*(\S+)", text)
    if m:
        k["file"] = m.group(1)
    m = re.search(r"Train:\s*\(?([\d,]+)", text)
    if m:
        k["train"] = int(m.group(1).replace(",", ""))
    m = re.search(r"EN İYİ MODEL:\s*(.+?)\s*\|\s*strateji:\s*(\S+)\s*\|\s*eşik:\s*([\d.]+)", text)
    if m:
        k["best_model"] = m.group(1); k["strategy"] = m.group(2); k["threshold"] = float(m.group(3))
    m = re.search(r"FINAL TEST → PR-AUC:\s*([\d.]+)\s*\|\s*Recall:\s*([\d.]+)\s*\|\s*Precision:\s*([\d.]+)", text)
    if m:
        k["test_prauc"] = float(m.group(1)); k["recall_pct"] = float(m.group(2)) * 100; k["precision_pct"] = float(m.group(3)) * 100
    return k


def load_deck(nb_path, prefix="", chapter_num=None):
    nb = read_notebook(nb_path)
    slides, cur, streams = [], None, []
    seen_h1 = False
    intro = None

    def finish():
        nonlocal cur
        if cur and (cur["visuals"] or cur["kind"] == "chapter"):
            slides.append(cur)
        cur = None

    def new(kind, title, body):
        t = strip_emoji(title)
        if kind == "chapter":
            t = re.sub(r"^BÖLÜM\s*(\d+)\s*[—–-]\s*", r"\1. ", t)
        m = re.match(r"^(\d+[a-z]?)\.\s*(.*)$", t)
        return {
            "kind": kind,
            "num": m.group(1) if m else "",
            "title": (m.group(2) if m else t).strip(),
            "subtitle": first_sentence(body),
            "notes_raw": body,
            "visuals": [],
        }

    for cell in nb.get("cells", []):
        src = join(cell.get("source", ""))
        if cell.get("cell_type") == "markdown":
            src = re.sub(r"^(\s*---+\s*\n)+", "", src)      # baştaki '---' ayraç satırlarını at
            first = src.split("\n", 1)[0].strip()
            body = src.split("\n", 1)[1] if "\n" in src else ""
            m = re.match(r"^(#{1,3})\s+(.*)", first)
            if m:
                level = len(m.group(1))
                if level == 1 and not seen_h1:
                    seen_h1 = True
                    t1 = strip_emoji(m.group(2))
                    t1 = re.split(r"\s[—–-]\s", t1)[-1].strip() or t1
                    intro = {"kind": "chapter", "num": chapter_num or "", "title": t1,
                             "subtitle": first_sentence(body), "visuals": [], "notes_raw": ""}
                    continue
                finish()
                cur = new("chapter" if level == 1 else "content", m.group(2), body)
            elif cur is not None:
                cur["notes_raw"] += "\n\n" + src
        elif cell.get("cell_type") == "code":
            for out in cell.get("outputs", []):
                ot = out.get("output_type")
                if ot == "stream":
                    streams.append(join(out.get("text", "")))
                    continue
                if ot not in ("display_data", "execute_result"):
                    continue
                data = out.get("data", {})
                vis = None
                if "image/png" in data:
                    b64 = re.sub(r"\s+", "", join(data["image/png"]))
                    vis = {"type": "img", "src": "data:image/png;base64," + b64}
                elif "text/html" in data and "<table" in join(data["text/html"]):
                    vis = {"type": "table", "html": clean_table_html(join(data["text/html"]))}
                if vis:
                    if cur is None:
                        cur = new("content", "Genel Bakış", "")
                    cur["visuals"].append(vis)

    # Son bölüm görselsiz bir "Özet" ise kapanış slaytı olarak tut
    if cur and not cur["visuals"] and cur["kind"] == "content" and "özet" in cur["title"].lower():
        cur["kind"] = "closing"
        slides.append(cur)
        cur = None
    finish()

    if intro and chapter_num and slides:        # 2. ve sonraki notebook'lar için bölüm ayracı slaytı
        slides.insert(0, intro)
    for s in slides:
        s["notes_html"] = md_to_html(s.pop("notes_raw"))
        if prefix and s["num"] and s["kind"] != "chapter":
            s["num"] = prefix + s["num"]

    kpi = parse_kpis("\n".join(streams))
    return {"slides": slides, "kpi": kpi}


def load_decks(paths):
    """Birden fazla notebook'u tek sunumda birleştirir (EDA + modelleme)."""
    all_slides, kpi = [], {}
    prefixes = ["", "M", "N", "O"]
    for i, p in enumerate(paths):
        d = load_deck(p, prefix=prefixes[min(i, 3)], chapter_num=str(i + 2) if i else None)
        sl = d["slides"]
        if i < len(paths) - 1:                   # ara notebook'ların "Özet" slaytı sunumun ortasında kalmasın
            sl = [x for x in sl if x["kind"] != "closing"]
        all_slides += sl
        kpi.update(d["kpi"])
    return {"slides": all_slides, "kpi": kpi}


# --------------------------------------------------------------------------
# 3) HTML şablonu
# --------------------------------------------------------------------------
TEMPLATE = r'''<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#070b1a;--bg2:#0d1630;--glass:rgba(255,255,255,.06);--line:rgba(255,255,255,.13);
--gold:#f5c451;--gold2:#ffe29a;--teal:#2dd4bf;--red:#ff4d6d;--txt:#e8ecf8;--mut:#9aa6c7}
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%;overflow:hidden;background:var(--bg);color:var(--txt);
font-family:"Segoe UI",Inter,system-ui,-apple-system,"Helvetica Neue",Arial,sans-serif}
#rain{position:fixed;inset:0;z-index:0;opacity:.75}
body:before{content:"";position:fixed;inset:0;z-index:0;pointer-events:none;
background:radial-gradient(900px 500px at 15% 10%,rgba(245,196,81,.10),transparent 60%),
radial-gradient(800px 600px at 90% 90%,rgba(45,212,191,.10),transparent 60%),
radial-gradient(700px 500px at 80% 5%,rgba(124,92,255,.12),transparent 60%)}
#progress{position:fixed;top:0;left:0;right:0;height:3px;z-index:50;background:rgba(255,255,255,.06)}
#progress i{display:block;height:100%;width:0;background:linear-gradient(90deg,var(--teal),var(--gold));
box-shadow:0 0 14px var(--gold);transition:width .6s cubic-bezier(.2,.8,.2,1)}
header.top{position:fixed;top:3px;left:0;right:0;height:58px;z-index:40;display:flex;align-items:center;
justify-content:space-between;padding:0 28px 0 24px;backdrop-filter:blur(10px);
background:linear-gradient(180deg,rgba(7,11,26,.85),rgba(7,11,26,0))}
.brand{display:flex;align-items:center;gap:12px}
.minicard{width:34px;height:23px;border-radius:5px;background:linear-gradient(135deg,var(--gold),#c98f1c);position:relative;
box-shadow:0 4px 14px rgba(245,196,81,.35)}
.minicard:after{content:"";position:absolute;left:5px;top:6px;width:9px;height:7px;border-radius:2px;background:rgba(7,11,26,.55)}
.brand b{letter-spacing:.18em;font-size:13px}
.brand small{display:block;color:var(--mut);font-size:10.5px;letter-spacing:.08em;margin-top:2px}
.hint{color:var(--mut);font-size:11.5px;letter-spacing:.04em}
.hint kbd{background:var(--glass);border:1px solid var(--line);border-radius:5px;padding:1px 6px;font-size:11px;color:var(--gold2)}
#dots{position:fixed;left:18px;top:50%;transform:translateY(-50%);z-index:40;display:flex;flex-direction:column;gap:10px}
#dots button{width:10px;height:10px;border-radius:50%;border:1px solid var(--line);background:transparent;cursor:pointer;
position:relative;transition:.3s}
#dots button:hover{transform:scale(1.5);border-color:var(--gold)}
#dots button.on{background:var(--gold);border-color:var(--gold);box-shadow:0 0 12px var(--gold);transform:scale(1.35)}
#dots button span{position:absolute;left:20px;top:50%;transform:translateY(-50%);white-space:nowrap;font-size:12px;
background:rgba(13,22,48,.95);border:1px solid var(--line);padding:4px 9px;border-radius:8px;opacity:0;pointer-events:none;transition:.2s}
#dots button:hover span{opacity:1}
#stage{position:fixed;inset:0;z-index:5;perspective:1600px}
.slide{position:absolute;inset:66px 0 46px 0;padding:14px 4.5vw 8px 64px;display:flex;flex-direction:column;opacity:0;
pointer-events:none;transform:translateX(7vw) scale(.97) rotateY(-5deg);filter:blur(8px);
transition:opacity .7s ease,transform .8s cubic-bezier(.2,.8,.2,1),filter .7s ease}
.slide.before{transform:translateX(-7vw) scale(.97) rotateY(5deg)}
.slide.active{opacity:1;transform:none;filter:none;pointer-events:auto}
.slide.active .reveal{animation:rise .8s cubic-bezier(.2,.8,.2,1) both;animation-delay:calc(var(--i,0)*110ms)}
@keyframes rise{from{opacity:0;transform:translateY(26px)}to{opacity:1;transform:none}}
.head{display:flex;align-items:center;gap:18px;margin-bottom:12px}
.badge-num{min-width:58px;height:58px;border-radius:16px;display:grid;place-items:center;font-weight:800;font-size:21px;
color:#1a1300;background:linear-gradient(135deg,var(--gold2),var(--gold));box-shadow:0 10px 30px rgba(245,196,81,.35);
letter-spacing:.02em}
.ttl{flex:1;min-width:0}
.ttl h2{font-size:clamp(22px,2.6vw,34px);font-weight:800;letter-spacing:-.01em;display:flex;align-items:center;gap:12px}
.ttl .ico{font-size:.95em;filter:drop-shadow(0 4px 10px rgba(0,0,0,.5))}
.ttl p{color:var(--mut);font-size:clamp(12.5px,1.2vw,15px);margin-top:4px;max-width:1100px;line-height:1.45;
display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
.chip-btn{background:var(--glass);border:1px solid var(--line);color:var(--txt);padding:9px 15px;border-radius:12px;cursor:pointer;
font-size:13px;transition:.25s;backdrop-filter:blur(8px);white-space:nowrap}
.chip-btn:hover{border-color:var(--gold);color:var(--gold2);transform:translateY(-2px)}
.viz{flex:1;min-height:0;display:flex}
.frame{position:relative;flex:1;min-height:0;border-radius:20px;background:#fff;overflow:hidden;display:flex;align-items:center;
justify-content:center;box-shadow:0 0 0 1px rgba(255,255,255,.18),0 30px 80px rgba(0,0,0,.55),0 0 60px rgba(245,196,81,.12)}
.frame:before{content:"";position:absolute;inset:0;border-radius:20px;padding:2px;pointer-events:none;z-index:4;
background:linear-gradient(120deg,var(--gold),transparent 30%,transparent 70%,var(--teal));
-webkit-mask:linear-gradient(#000 0 0) content-box,linear-gradient(#000 0 0);-webkit-mask-composite:xor;mask-composite:exclude}
.vis{width:100%;height:100%;display:flex;align-items:center;justify-content:center;padding:10px}
.vis img{max-width:100%;max-height:100%;object-fit:contain;cursor:zoom-in;animation:pop .7s cubic-bezier(.2,.8,.2,1) both}
@keyframes pop{from{opacity:0;transform:scale(.96)}to{opacity:1;transform:none}}
.tablewrap{width:100%;height:100%;overflow:auto;color:#14203f;padding:6px 10px}
.tablewrap table{border-collapse:collapse;font-size:clamp(11px,1.05vw,14px);margin:auto}
.tablewrap th{position:sticky;top:0;background:#0d1630;color:var(--gold2);padding:8px 12px;text-align:right;font-weight:600;white-space:nowrap}
.tablewrap td{padding:7px 12px;text-align:right;border-bottom:1px solid #e6e9f4;font-variant-numeric:tabular-nums;white-space:nowrap}
.tablewrap th:first-child,.tablewrap td:first-child{text-align:left;font-weight:700;background:#f4f6fc}
.tablewrap tr:hover td{background:#fff7dd}
.beam{position:absolute;top:0;bottom:0;width:90px;left:-120px;z-index:3;pointer-events:none;opacity:0;
background:linear-gradient(90deg,transparent,rgba(45,212,191,.55),rgba(245,196,81,.55),transparent);filter:blur(2px)}
.frame.scanning .beam{opacity:1;animation:sweep 1.5s ease-in-out}
@keyframes sweep{from{left:-120px}to{left:calc(100% + 30px)}}
.status{position:absolute;top:12px;right:14px;z-index:5;font-size:11px;letter-spacing:.14em;font-weight:700;padding:6px 11px;
border-radius:999px;background:rgba(13,22,48,.9);color:var(--teal);border:1px solid rgba(45,212,191,.5)}
.status:before{content:"";display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:8px;background:currentColor;
animation:blink 1s infinite}
.status.done{color:#34d399;border-color:rgba(52,211,153,.5)}
.status.done:before{animation:none}
@keyframes blink{50%{opacity:.2}}
.thumbs{display:flex;gap:8px;justify-content:center;margin-top:12px;flex-wrap:wrap}
.thumbs button{background:var(--glass);border:1px solid var(--line);color:var(--mut);padding:6px 14px;border-radius:999px;cursor:pointer;
font-size:12.5px;transition:.25s}
.thumbs button:hover{color:var(--txt);border-color:var(--gold)}
.thumbs button.on{background:linear-gradient(135deg,var(--gold2),var(--gold));color:#1a1300;border-color:transparent;font-weight:700}
/* ---------------- HERO ---------------- */
.hero{flex-direction:row;align-items:center;gap:3vw;padding-left:84px}
.hero-left{flex:1.15;min-width:0}
.eyebrow{color:var(--gold);letter-spacing:.3em;font-size:12px;font-weight:700;margin-bottom:16px}
.hero h1{font-size:clamp(34px,5vw,68px);line-height:1.04;font-weight:900;letter-spacing:-.02em;min-height:2.1em}
.hero h1 .caret{display:inline-block;width:.07em;height:.9em;background:var(--gold);margin-left:6px;vertical-align:-.08em;animation:blink .9s infinite}
.hero h1 em{font-style:normal;background:linear-gradient(90deg,var(--gold2),var(--gold),var(--teal));-webkit-background-clip:text;
background-clip:text;color:transparent}
.lead{color:var(--mut);font-size:clamp(14px,1.4vw,18px);margin:14px 0 26px;max-width:640px;line-height:1.5}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:12px;max-width:640px}
.kpi{background:var(--glass);border:1px solid var(--line);border-radius:16px;padding:14px 16px;backdrop-filter:blur(8px);position:relative;overflow:hidden}
.kpi:after{content:"";position:absolute;left:0;bottom:0;height:3px;width:100%;background:linear-gradient(90deg,var(--gold),var(--teal));opacity:.8}
.kpi b{display:block;font-size:clamp(22px,2.4vw,32px);font-weight:800;color:var(--gold2);font-variant-numeric:tabular-nums}
.kpi span{font-size:11.5px;color:var(--mut);letter-spacing:.06em}
.empty{margin-top:22px;padding:14px 18px;border:1px dashed var(--gold);border-radius:14px;color:var(--gold2);max-width:640px;line-height:1.5;font-size:14px}
.hero-right{flex:1;display:flex;flex-direction:column;align-items:center;gap:26px;min-width:0}
.card3d{perspective:1100px;animation:floaty 5s ease-in-out infinite}
@keyframes floaty{50%{transform:translateY(-14px)}}
.card-inner{width:min(430px,36vw);aspect-ratio:1.586;border-radius:24px;position:relative;overflow:hidden;padding:5.2%;display:flex;flex-direction:column;
justify-content:space-between;background:linear-gradient(135deg,#1d2f6b,#0b1233 55%,#35206b);
box-shadow:0 40px 90px rgba(0,0,0,.65),0 0 0 1px rgba(255,255,255,.14) inset,0 0 70px rgba(245,196,81,.18);
transform:rotateX(var(--rx,8deg)) rotateY(var(--ry,-14deg));transition:transform .18s ease-out}
.holo{position:absolute;inset:0;background:linear-gradient(115deg,transparent 20%,rgba(255,255,255,.14) 35%,rgba(45,212,191,.16) 45%,
rgba(245,196,81,.18) 55%,transparent 70%);background-size:250% 250%;animation:holo 4.5s linear infinite;mix-blend-mode:screen}
@keyframes holo{from{background-position:130% 0}to{background-position:-130% 0}}
.cbeam{position:absolute;left:0;right:0;height:60px;top:-70px;background:linear-gradient(180deg,transparent,rgba(45,212,191,.45),transparent);
animation:cscan 3.6s ease-in-out infinite}
@keyframes cscan{0%{top:-70px}100%{top:110%}}
.crow{display:flex;justify-content:space-between;align-items:center;position:relative;z-index:2}
.crow .bank{font-weight:800;letter-spacing:.22em;font-size:clamp(10px,1.1vw,14px)}
.chip{width:15%;aspect-ratio:1.28;border-radius:9px;position:relative;z-index:2;
background:linear-gradient(135deg,#ffe29a,#e0a82e 55%,#b9821a);box-shadow:0 2px 8px rgba(0,0,0,.4) inset}
.chip:before,.chip:after{content:"";position:absolute;left:0;right:0;height:1px;background:rgba(0,0,0,.35)}
.chip:before{top:33%}.chip:after{top:66%}
.cnum{font-family:"Consolas","Courier New",monospace;letter-spacing:.14em;font-size:clamp(14px,1.9vw,25px);position:relative;z-index:2;
text-shadow:0 2px 10px rgba(0,0,0,.6)}
.cbot{display:flex;justify-content:space-between;align-items:flex-end;position:relative;z-index:2;gap:10px}
.cbot small{display:block;color:var(--mut);font-size:clamp(7px,.75vw,10px);letter-spacing:.14em;margin-bottom:3px}
.cbot b{font-size:clamp(9px,1vw,13px);letter-spacing:.08em}
.cstatus{position:absolute;right:5%;top:30%;z-index:3;font-weight:800;font-size:clamp(9px,1vw,13px);letter-spacing:.14em;padding:5px 11px;border-radius:8px;
border:2px solid;transform:rotate(-8deg);transition:.4s}
.cstatus.ok{color:#34d399;border-color:#34d399;background:rgba(52,211,153,.12)}
.cstatus.bad{color:var(--red);border-color:var(--red);background:rgba(255,77,109,.14);animation:shake .5s}
@keyframes shake{25%{translate:-4px 0}75%{translate:4px 0}}
.donutbox{display:flex;align-items:center;gap:20px;background:var(--glass);border:1px solid var(--line);border-radius:20px;padding:14px 22px;backdrop-filter:blur(10px)}
.donut{width:110px;height:110px;transform:rotate(-90deg)}
.donut circle{fill:none;stroke-width:13}
.donut .bg{stroke:rgba(255,255,255,.1)}
.donut .nm{stroke:var(--teal)}
.donut .fr{stroke:var(--red);stroke-linecap:round;stroke-dasharray:0 400;transition:stroke-dasharray 1.8s cubic-bezier(.2,.8,.2,1) .4s;filter:drop-shadow(0 0 6px var(--red))}
.legend{font-size:12.5px;line-height:1.9;color:var(--mut)}
.legend b{color:var(--txt);font-variant-numeric:tabular-nums}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:8px}
/* ---------------- AGENDA / CHAPTER / CLOSING ---------------- */
.agenda-grid{flex:1;min-height:0;overflow:auto;display:grid;grid-template-columns:repeat(auto-fill,minmax(235px,1fr));gap:14px;padding:6px 6px 16px;align-content:start}
.acard{background:var(--glass);border:1px solid var(--line);border-radius:18px;padding:16px;cursor:pointer;text-align:left;color:var(--txt);
transition:.3s;position:relative;overflow:hidden;backdrop-filter:blur(8px)}
.acard:before{content:"";position:absolute;inset:0;background:linear-gradient(135deg,rgba(245,196,81,.16),transparent 55%);opacity:0;transition:.3s}
.acard:hover{transform:translateY(-6px) rotate(-.4deg);border-color:var(--gold);box-shadow:0 18px 40px rgba(0,0,0,.45)}
.acard:hover:before{opacity:1}
.acard .n{font-size:11px;color:var(--gold);letter-spacing:.2em;font-weight:700}
.acard .i{font-size:30px;margin:8px 0}
.acard h4{font-size:15px;line-height:1.3;position:relative}
.acard p{font-size:11.5px;color:var(--mut);margin-top:6px;position:relative}
.chapter{align-items:center;justify-content:center;text-align:center}
.chapter .big{font-size:clamp(70px,13vw,190px);font-weight:900;line-height:1;color:transparent;-webkit-text-stroke:2px rgba(245,196,81,.6);letter-spacing:-.03em}
.chapter h2{font-size:clamp(28px,4.4vw,58px);font-weight:900;margin-top:-.2em}
.chapter p{color:var(--mut);max-width:760px;margin:18px auto 0;font-size:clamp(14px,1.4vw,18px);line-height:1.55}
.chapter .rule{width:140px;height:4px;border-radius:4px;margin:22px auto 0;background:linear-gradient(90deg,var(--teal),var(--gold));box-shadow:0 0 18px var(--gold)}
.closing{align-items:center;justify-content:flex-start;overflow:auto}
.closing .notes-inline{max-width:980px;width:100%;margin-top:14px}
/* ---------------- NOTES PANEL ---------------- */
#notes{position:fixed;top:66px;right:0;bottom:46px;width:min(500px,92vw);z-index:60;transform:translateX(105%);transition:transform .55s cubic-bezier(.2,.8,.2,1);
background:rgba(10,16,38,.94);backdrop-filter:blur(16px);border-left:1px solid var(--line);padding:22px 24px;overflow:auto;box-shadow:-30px 0 80px rgba(0,0,0,.5)}
#notes.open{transform:none}
#notes h3{color:var(--gold);letter-spacing:.2em;font-size:12px;margin-bottom:12px}
#notes .x{position:absolute;right:16px;top:14px;background:none;border:none;color:var(--mut);font-size:22px;cursor:pointer}
.notes-body,.notes-inline{font-size:14.5px;line-height:1.65;color:#d6dcf0}
.notes-body p,.notes-inline p{margin-bottom:10px}
.notes-body ul,.notes-body ol,.notes-inline ul,.notes-inline ol{margin:0 0 12px 20px}
.notes-body li,.notes-inline li{margin-bottom:5px}
.notes-body code,.notes-inline code{background:rgba(245,196,81,.14);color:var(--gold2);padding:1px 6px;border-radius:6px;font-size:.92em}
.notes-body blockquote,.notes-inline blockquote{border-left:3px solid var(--gold);padding:8px 14px;margin:10px 0;background:var(--glass);border-radius:0 10px 10px 0}
.mdtable{border-collapse:collapse;width:100%;font-size:12.5px;margin:8px 0 14px}
.mdtable th{background:rgba(245,196,81,.16);color:var(--gold2);text-align:left;padding:7px 9px}
.mdtable td{padding:7px 9px;border-bottom:1px solid var(--line);vertical-align:top}
/* ---------------- FOOTER / CONTROLS ---------------- */
.ticker{position:fixed;left:0;right:0;bottom:0;height:40px;z-index:45;display:flex;align-items:center;overflow:hidden;
background:rgba(7,11,26,.92);border-top:1px solid var(--line);font-family:"Consolas","Courier New",monospace;font-size:12px}
.ticker .tag{flex:none;padding:0 14px;height:100%;display:flex;align-items:center;gap:8px;background:linear-gradient(90deg,#b9821a,var(--gold));
color:#1a1300;font-weight:800;letter-spacing:.12em;font-size:11px;font-family:inherit}
.track{display:flex;gap:44px;white-space:nowrap;animation:marq 70s linear infinite;padding-left:30px}
@keyframes marq{to{transform:translateX(-50%)}}
.tx b.ok{color:#34d399}.tx b.bad{color:var(--red)}.tx{color:var(--mut)}
.right-info{flex:none;display:flex;gap:16px;align-items:center;padding:0 16px;height:100%;background:rgba(7,11,26,.96);border-left:1px solid var(--line);color:var(--mut);font-size:11.5px}
.live.on{color:#34d399}.live.off{color:var(--mut)}
#ctr{color:var(--gold2);font-weight:700;letter-spacing:.1em}
.ctrls{position:fixed;right:18px;bottom:56px;z-index:46;display:flex;gap:8px}
.ctrls button{width:40px;height:40px;border-radius:12px;border:1px solid var(--line);background:rgba(13,22,48,.85);color:var(--txt);cursor:pointer;font-size:16px;
transition:.25s;backdrop-filter:blur(8px)}
.ctrls button:hover{border-color:var(--gold);color:var(--gold2);transform:translateY(-3px)}
#lightbox{position:fixed;inset:0;z-index:100;background:rgba(3,6,16,.93);display:none;align-items:center;justify-content:center;padding:3vh 3vw;cursor:zoom-out}
#lightbox.on{display:flex;animation:pop .35s both}
#lightbox img{max-width:100%;max-height:100%;border-radius:14px;background:#fff;box-shadow:0 0 80px rgba(245,196,81,.25)}
#toast{position:fixed;top:78px;left:50%;transform:translate(-50%,-30px);opacity:0;z-index:120;background:linear-gradient(135deg,var(--gold2),var(--gold));color:#1a1300;
padding:11px 20px;border-radius:14px;font-weight:700;font-size:13.5px;box-shadow:0 14px 40px rgba(0,0,0,.5);transition:.5s;pointer-events:none}
#toast.on{opacity:1;transform:translate(-50%,0)}
@media (max-width:900px){.hero{flex-direction:column;overflow:auto;padding-left:44px}.card-inner{width:min(340px,80vw)}.hint{display:none}
.slide{padding-left:40px}}
@media (prefers-reduced-motion:reduce){*{animation-duration:.01ms!important;transition-duration:.01ms!important}}
</style>
</head>
<body>
<canvas id="rain"></canvas>
<div id="progress"><i></i></div>
<header class="top">
  <div class="brand"><div class="minicard"></div><div><b>FRAUDSHIELD ANALYTICS</b><small>BANKACILIK GÜVENLİĞİ · YAPAY ZEKA LABORATUVARI</small></div></div>
  <div class="hint"><kbd>←</kbd> <kbd>→</kbd> slayt &nbsp; <kbd>↑</kbd> <kbd>↓</kbd> grafik &nbsp; <kbd>N</kbd> notlar &nbsp; <kbd>G</kbd> menü &nbsp; <kbd>F</kbd> tam ekran</div>
</header>
<nav id="dots"></nav>
<main id="stage"></main>
<aside id="notes"><button class="x" data-act="notes" aria-label="Kapat">×</button><h3>ANLATIM NOTLARI</h3><div class="notes-body" id="notesBody"></div></aside>
<div class="ctrls">
  <button data-act="prev" title="Önceki">◀</button><button data-act="next" title="Sonraki">▶</button>
  <button data-act="notes" title="Notlar (N)">📝</button><button data-act="menu" title="Menü (G)">▦</button><button data-act="full" title="Tam ekran (F)">⛶</button>
</div>
<footer class="ticker">
  <div class="tag">● SİMÜLASYON</div>
  <div style="flex:1;overflow:hidden"><div class="track" id="track"></div></div>
  <div class="right-info"><span class="live off" id="live">○ STATİK</span><span>Güncelleme: __BUILT__</span><span id="ctr">01 / 01</span></div>
</footer>
<div id="lightbox"><img alt=""></div>
<div id="toast"></div>
<script id="data" type="application/json">__DATA_JSON__</script>
<script>
(function(){
'use strict';
var D=JSON.parse(document.getElementById('data').textContent);
var $=function(s,r){return (r||document).querySelector(s)};
var $$=function(s,r){return Array.prototype.slice.call((r||document).querySelectorAll(s))};
var esc=function(s){return String(s).replace(/[&<>"']/g,function(c){return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]})};
var ICONS=[[/sınıf|hedef/,'🎯'],[/histogram/,'📈'],[/boxplot|kutu/,'📦'],[/scatter|saçılım|pairplot/,'🔵'],[/korelasyon/,'🔗'],
[/sızıntı|aşırı güçlü/,'🚨'],[/kategorik/,'🏷️'],[/zaman/,'⏰'],[/eksik|tekrar/,'🧩'],[/istatistik|özet/,'📊'],[/yükleme|ilk bakış/,'📥'],
[/encoding/,'🔢'],[/ölçekleme/,'⚖️'],[/özellik seçimi/,'✂️'],[/özellik üretimi/,'🧪'],[/train|test/,'🧬'],[/kaydet/,'💾'],[/temizlik/,'🧹']];
function iconFor(t){t=t.toLocaleLowerCase('tr');for(var i=0;i<ICONS.length;i++){if(ICONS[i][0].test(t))return ICONS[i][1]}return '💳'}
var pad=function(n){return (n<10?'0':'')+n};
var fmt=function(n,d){return Number(n).toLocaleString('tr-TR',{minimumFractionDigits:d||0,maximumFractionDigits:d||0})};

/* ------------ slayt listesi ------------ */
var K=D.kpi||{};
var content=D.slides.filter(function(s){return s.kind==='content'&&s.visuals.length});
var order=[{kind:'hero'}];
if(content.length>2)order.push({kind:'agenda'});
D.slides.forEach(function(s){order.push(s)});
var stage=$('#stage'),els=[],idx=0,visIdx={},typedTimer=null;

function buildHero(){
  var el=document.createElement('section');el.className='slide hero';
  var kp=[];
  if(K.rows!=null)kp.push(['İşlem sayısı',K.rows,0,'']);
  if(K.cols!=null)kp.push(['Sütun',K.cols,0,'']);
  if(K.fraud_pct!=null)kp.push(['Fraud oranı',K.fraud_pct,2,'%']);
  if(K.selected!=null)kp.push(['Seçilen özellik',K.selected,0,'']);
  if(K.best_model)kp.push(['En iyi model',K.best_model,0,'','str']);
  if(K.test_prauc!=null)kp.push(['Test PR-AUC',K.test_prauc,3,'']);
  if(K.recall_pct!=null)kp.push(['Fraud yakalama (Recall)',K.recall_pct,1,'%']);
  var kHtml=kp.map(function(k){var i=3+kp.indexOf(k);
    if(k[4]==='str')return '<div class="kpi reveal" style="--i:'+i+'"><b style="font-size:clamp(15px,1.6vw,21px)">'+esc(k[1])+'</b><span>'+esc(k[0]).toUpperCase()+'</span></div>';
    return '<div class="kpi reveal" style="--i:'+i+'"><b data-count="'+k[1]+'" data-dec="'+k[2]+'" data-suf="'+k[3]+'">0</b><span>'+esc(k[0]).toUpperCase()+'</span></div>'}).join('');
  var empty=D.slides.length?'':'<div class="empty">Henüz grafik yok. Jupyter\'de <b>Run → Run All Cells</b> çalıştır, sonra <b>Ctrl+S</b> ile kaydet. Bu sayfa kendiliğinden güncellenecek.</div>';
  var donut='';
  if(K.fraud_pct!=null){
    var C=2*Math.PI*45,fl=C*K.fraud_pct/100;
    donut='<div class="donutbox reveal" style="--i:5"><svg class="donut" viewBox="0 0 110 110"><circle class="bg" cx="55" cy="55" r="45"/><circle class="nm" cx="55" cy="55" r="45" stroke-dasharray="'+C+' '+C+'" style="opacity:.55"/><circle class="fr" cx="55" cy="55" r="45" data-len="'+fl+'" data-c="'+C+'"/></svg>'+
    '<div class="legend"><div><span class="dot" style="background:var(--teal)"></span>Normal <b>'+(K.normal!=null?fmt(K.normal):'—')+'</b></div><div><span class="dot" style="background:var(--red)"></span>Fraud <b>'+(K.fraud!=null?fmt(K.fraud):'—')+'</b></div><div style="font-size:11px">Sınıf dağılımı</div></div></div>';
  }
  el.innerHTML='<div class="hero-left"><p class="eyebrow reveal" style="--i:0">YAPAY ZEKA PROJESİ · BANKACILIK GÜVENLİĞİ</p>'+
  '<h1 class="reveal" style="--i:1"><span id="typed"></span><span class="caret"></span></h1>'+
  '<p class="lead reveal" style="--i:2">'+(K.best_model?'Veri analizi, özellik mühendisliği ve modelleme: dolandırıcılığı hangi model, hangi eşikle, ne kadar güvenilir yakalıyor?':'Keşifsel veri analizi, özellik mühendisliği ve ön işleme. Modelden önce veriyi tanıyoruz: dolandırıcılık nerede, ne zaman ve nasıl ortaya çıkıyor?')+'</p>'+
  '<div class="kpis">'+kHtml+'</div>'+empty+'</div>'+
  '<div class="hero-right"><div class="card3d reveal" style="--i:2"><div class="card-inner" id="card"><div class="holo"></div><div class="cbeam"></div>'+
  '<div class="crow"><span class="bank">FRAUDSHIELD BANK</span><svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="#e8ecf8" stroke-width="1.8" stroke-linecap="round"><path d="M8 8a6 6 0 0 1 0 8"/><path d="M12 5a10 10 0 0 1 0 14"/><path d="M16 2a14 14 0 0 1 0 20" opacity=".6"/></svg></div>'+
  '<div class="chip"></div><div class="cnum">4532 8891 •••• 0172</div><div class="cstatus ok" id="cstatus">ONAYLANDI ✓</div>'+
  '<div class="cbot"><div><small>KART SAHİBİ</small><b>FRAUD DETECTION LAB</b></div><div><small>HEDEF</small><b>'+esc(K.target||'Fraud_Label')+'</b></div><div><small>VERİ</small><b>'+esc(K.file||'dataset.csv')+'</b></div></div></div></div>'+donut+'</div>';
  return el;
}
function buildAgenda(){
  var el=document.createElement('section');el.className='slide agenda';
  var cards=order.map(function(s,i){return {s:s,i:i}}).filter(function(o){return o.s.kind==='content'&&o.s.visuals.length}).map(function(o,n){
    return '<button class="acard reveal" style="--i:'+Math.min(n,12)+'" data-go="'+o.i+'"><div class="n">'+(o.s.num?'BÖLÜM '+esc(o.s.num):'KONU '+pad(n+1))+'</div><div class="i">'+iconFor(o.s.title)+'</div><h4>'+esc(o.s.title)+'</h4><p>'+o.s.visuals.length+' görsel</p></button>'}).join('');
  el.innerHTML='<div class="head reveal" style="--i:0"><div class="badge-num">▦</div><div class="ttl"><h2>Sunum Menüsü</h2><p>Bir konuya tıklayarak doğrudan o grafiğe git.</p></div></div><div class="agenda-grid">'+cards+'</div>';
  return el;
}
function buildChapter(s){
  var el=document.createElement('section');el.className='slide chapter';
  el.innerHTML='<div class="big reveal" style="--i:0">'+esc(s.num||'§')+'</div><h2 class="reveal" style="--i:1">'+esc(s.title)+'</h2><div class="rule reveal" style="--i:2"></div>'+(s.subtitle?'<p class="reveal" style="--i:3">'+esc(s.subtitle)+'</p>':'');
  return el;
}
function buildClosing(s){
  var el=document.createElement('section');el.className='slide closing';
  el.innerHTML='<div class="head reveal" style="--i:0"><div class="badge-num">✓</div><div class="ttl"><h2><span class="ico">🏁</span>'+esc(s.title)+'</h2></div></div><div class="notes-inline reveal" style="--i:1">'+s.notes_html+'</div>';
  return el;
}
function buildContent(s,si){
  var el=document.createElement('section');el.className='slide content';
  var th=s.visuals.length>1?s.visuals.map(function(v,k){return '<button data-vis="'+k+'" class="'+(k===0?'on':'')+'">'+(v.type==='table'?'▦ Tablo ':'📊 Grafik ')+(k+1)+'</button>'}).join(''):'';
  el.innerHTML='<div class="head reveal" style="--i:0"><div class="badge-num">'+esc(s.num||'•')+'</div><div class="ttl"><h2><span class="ico">'+iconFor(s.title)+'</span>'+esc(s.title)+'</h2>'+(s.subtitle?'<p>'+esc(s.subtitle)+'</p>':'')+'</div>'+
  '<button class="chip-btn" data-act="notes">📝 Anlatım notları</button></div>'+
  '<div class="viz reveal" style="--i:1"><div class="frame"><div class="beam"></div><div class="status">TARANIYOR…</div><div class="vis"></div></div></div>'+
  (th?'<div class="thumbs reveal" style="--i:2">'+th+'</div>':'');
  visIdx[si]=0;
  return el;
}
order.forEach(function(s,i){
  var el=s.kind==='hero'?buildHero():s.kind==='agenda'?buildAgenda():s.kind==='chapter'?buildChapter(s):s.kind==='closing'?buildClosing(s):buildContent(s,i);
  stage.appendChild(el);els.push(el);
});

/* ------------ görsel gösterme + tarama animasyonu ------------ */
function showVis(i,k){
  var s=order[i],el=els[i],v=s.visuals[k];if(!v)return;
  visIdx[i]=k;
  var vis=$('.vis',el),fr=$('.frame',el),st=$('.status',el);
  if(v.type==='img'){vis.innerHTML='<img alt="">';$('img',vis).src=v.src;$('img',vis).onclick=function(){openLB(v.src)}}
  else{vis.innerHTML='<div class="tablewrap">'+v.html+'</div>'}
  $$('.thumbs button',el).forEach(function(b,n){b.classList.toggle('on',n===k)});
  fr.classList.remove('scanning');void fr.offsetWidth;fr.classList.add('scanning');
  st.className='status';st.textContent='TARANIYOR…';
  clearTimeout(el._t);el._t=setTimeout(function(){st.className='status done';st.textContent='ANALİZ TAMAM ✓';fr.classList.remove('scanning')},1500);
}
function openLB(src){$('#lightbox img').src=src;$('#lightbox').classList.add('on')}
$('#lightbox').onclick=function(){this.classList.remove('on')};

/* ------------ hero animasyonları ------------ */
function heroOn(){
  var el=els[0],t=$('#typed',el);if(!t)return;
  var parts=['Kredi Kartı ','Dolandırıcılık ','Tespiti'],full=parts.join(''),n=0;
  clearInterval(typedTimer);t.innerHTML='';
  typedTimer=setInterval(function(){n++;var s=full.slice(0,n);t.innerHTML=esc(s.slice(0,15))+(s.length>15?'<em>'+esc(s.slice(15))+'</em>':'');if(n>=full.length)clearInterval(typedTimer)},55);
  $$('[data-count]',el).forEach(function(b){
    var to=parseFloat(b.dataset.count),dec=parseInt(b.dataset.dec,10)||0,suf=b.dataset.suf||'',t0=null;
    function step(ts){if(!t0)t0=ts;var p=Math.min(1,(ts-t0)/1500),e=1-Math.pow(1-p,3);b.textContent=fmt(to*e,dec)+suf;if(p<1)requestAnimationFrame(step)}
    requestAnimationFrame(step);
  });
  var fr=$('.fr',el);if(fr){fr.style.strokeDasharray='0 400';setTimeout(function(){fr.style.strokeDasharray=fr.dataset.len+' '+fr.dataset.c},80)}
}
(function cardLoop(){
  var st=$('#cstatus');if(!st)return;var bad=false;
  setInterval(function(){bad=!bad&&Math.random()>.35;st.className='cstatus '+(bad?'bad':'ok');st.textContent=bad?'ENGELLENDİ ⚠':'ONAYLANDI ✓'},3600);
  var hero=els[0],card=$('#card');
  hero.addEventListener('mousemove',function(e){var r=hero.getBoundingClientRect(),x=(e.clientX-r.left)/r.width-.5,y=(e.clientY-r.top)/r.height-.5;
    card.style.setProperty('--ry',(x*34)+'deg');card.style.setProperty('--rx',(-y*26)+'deg')});
})();

/* ------------ gezinme ------------ */
var dots=$('#dots');
order.forEach(function(s,i){
  var b=document.createElement('button');
  var label=s.kind==='hero'?'Başlangıç':s.kind==='agenda'?'Menü':s.title;
  b.innerHTML='<span>'+esc(label)+'</span>';b.onclick=function(){go(i)};dots.appendChild(b);
});
function hasNotes(s){return s.notes_html&&s.notes_html.length>0}
function go(n,silent){
  n=Math.max(0,Math.min(order.length-1,n));
  els.forEach(function(el,i){el.classList.toggle('active',i===n);el.classList.toggle('before',i<n)});
  $$('#dots button').forEach(function(b,i){b.classList.toggle('on',i===n)});
  $('#progress i').style.width=(order.length>1?n/(order.length-1)*100:100)+'%';
  $('#ctr').textContent=pad(n+1)+' / '+pad(order.length);
  idx=n;
  var s=order[n];
  $('#notesBody').innerHTML=hasNotes(s)?s.notes_html:'<p>Bu slayt için not yok.</p>';
  if(s.kind==='hero')heroOn();
  if(s.kind==='content')showVis(n,visIdx[n]||0);
  try{history.replaceState(null,'','#'+(n+1))}catch(e){}
}
function next(){go(idx+1)}function prev(){go(idx-1)}
function visStep(d){var s=order[idx];if(s.kind!=='content'||s.visuals.length<2)return;showVis(idx,(visIdx[idx]+d+s.visuals.length)%s.visuals.length)}
function toggleNotes(){$('#notes').classList.toggle('open')}
function menu(){var a=order.findIndex(function(s){return s.kind==='agenda'});go(a>=0?a:0)}
function full(){var d=document;if(!d.fullscreenElement){(d.documentElement.requestFullscreen||function(){}).call(d.documentElement)}else{d.exitFullscreen()}}
var ACT={prev:prev,next:next,notes:toggleNotes,menu:menu,full:full};
document.addEventListener('click',function(e){
  var a=e.target.closest('[data-act]');if(a){ACT[a.dataset.act]();return}
  var g=e.target.closest('[data-go]');if(g){go(parseInt(g.dataset.go,10));return}
  var v=e.target.closest('[data-vis]');if(v){showVis(idx,parseInt(v.dataset.vis,10))}
});
document.addEventListener('keydown',function(e){
  if(e.ctrlKey||e.metaKey||e.altKey)return;
  var k=e.key;
  if(k==='ArrowRight'||k==='PageDown'||k===' '){e.preventDefault();next()}
  else if(k==='ArrowLeft'||k==='PageUp'){e.preventDefault();prev()}
  else if(k==='ArrowDown'){e.preventDefault();visStep(1)}
  else if(k==='ArrowUp'){e.preventDefault();visStep(-1)}
  else if(k==='Home')go(0);else if(k==='End')go(order.length-1);
  else if(k==='n'||k==='N')toggleNotes();
  else if(k==='g'||k==='G')menu();
  else if(k==='f'||k==='F')full();
  else if(k==='Escape'){$('#lightbox').classList.remove('on');$('#notes').classList.remove('open')}
});

/* ------------ simülasyon şeridi ------------ */
(function(){
  var types=['POS','Online','ATM','Havale'],h='';
  for(var r=0;r<2;r++){for(var i=0;i<26;i++){
    var amt=(Math.random()*Math.random()*900+3).toFixed(2),bad=Math.random()<.13;
    h+='<span class="tx">TXN_'+(10000+Math.floor(Math.random()*89999))+' · $'+amt+' · '+types[i%4]+' · <b class="'+(bad?'bad':'ok')+'">'+(bad?'ENGELLENDİ ⚠':'ONAYLANDI ✓')+'</b></span>';
  }}
  $('#track').innerHTML=h;
})();

/* ------------ arka plan: işlem yağmuru ------------ */
(function(){
  try{
    var c=$('#rain'),x=c.getContext('2d');if(!x)return;
    var w,h,cols,drops,G='01$€₺%¥£#'.split('');
    function size(){w=c.width=innerWidth;h=c.height=innerHeight;cols=Math.floor(w/26);drops=[];for(var i=0;i<cols;i++)drops.push(Math.random()*h/26)}
    size();addEventListener('resize',size);
    if(matchMedia&&matchMedia('(prefers-reduced-motion: reduce)').matches)return;
    setInterval(function(){
      x.fillStyle='rgba(7,11,26,.13)';x.fillRect(0,0,w,h);x.font='16px monospace';
      for(var i=0;i<cols;i++){
        x.fillStyle=Math.random()>.97?'rgba(245,196,81,.6)':'rgba(45,212,191,.2)';
        x.fillText(G[Math.floor(Math.random()*G.length)],i*26,drops[i]*26);
        if(drops[i]*26>h&&Math.random()>.975)drops[i]=0;drops[i]+=.5;
      }
    },55);
  }catch(e){}
})();

/* ------------ canlı senkron: notebook değişince sayfa yenilenir ------------ */
function toast(m){var t=$('#toast');t.textContent=m;t.classList.add('on');setTimeout(function(){t.classList.remove('on')},3800)}
function poll(){
  fetch('/__version',{cache:'no-store'}).then(function(r){if(!r.ok)throw 0;return r.text()}).then(function(v){
    var l=$('#live');l.className='live on';l.textContent='● CANLI';
    if(v.trim()!==String(D.version)){try{sessionStorage.setItem('slide',idx);sessionStorage.setItem('flash','1')}catch(e){}location.reload()}
  }).catch(function(){var l=$('#live');l.className='live off';l.textContent='○ STATİK'});
}
setInterval(poll,1500);poll();

/* ------------ başlangıç ------------ */
var start=0;
try{var ss=sessionStorage.getItem('slide');if(ss!==null){start=parseInt(ss,10)||0;sessionStorage.removeItem('slide')}
else if(location.hash){start=(parseInt(location.hash.slice(1),10)||1)-1}
if(sessionStorage.getItem('flash')){sessionStorage.removeItem('flash');setTimeout(function(){toast('Notebook güncellendi · yeni grafikler yüklendi ✓')},700)}}catch(e){}
go(start);
window.__deck={go:go,next:next,prev:prev,order:order};
})();
</script>
</body>
</html>
'''


def render_html(deck, version, title):
    data = {
        "version": version,
        "slides": deck["slides"],
        "kpi": deck["kpi"],
    }
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    page = TEMPLATE.replace("__DATA_JSON__", payload)
    page = page.replace("__TITLE__", htmllib.escape(title))
    page = page.replace("__BUILT__", time.strftime("%H:%M:%S"))
    return page


# --------------------------------------------------------------------------
# 4) Sunucu + dosya izleyici
# --------------------------------------------------------------------------
STATE = {"html": b"", "version": 1}
LOCK = threading.Lock()


def build(nb_paths, out_path, title):
    deck = load_decks(nb_paths)
    with LOCK:
        STATE["version"] += 1 if STATE["html"] else 0
        page = render_html(deck, STATE["version"], title)
        STATE["html"] = page.encode("utf-8")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(page)
    n_vis = sum(len(s["visuals"]) for s in deck["slides"])
    print("[%s] sunum güncellendi: %d slayt, %d görsel" % (time.strftime("%H:%M:%S"), len(deck["slides"]), n_vis), flush=True)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        with LOCK:
            html_bytes, version = STATE["html"], STATE["version"]
        if path == "/__version":
            self._send(str(version).encode(), "text/plain; charset=utf-8")
        elif path in ("/", "/index.html", "/sunum.html"):
            self._send(html_bytes, "text/html; charset=utf-8")
        else:
            self.send_error(404)


def mtimes(paths):
    out = []
    for p in paths:
        try:
            out.append(os.stat(p).st_mtime_ns)
        except OSError:
            out.append(0)
    return tuple(out)


def watch(nb_paths, out_path, title):
    last = mtimes(nb_paths)
    while True:
        time.sleep(1.0)
        cur = mtimes(nb_paths)
        if cur != last:
            time.sleep(0.8)                  # Jupyter'in yazmayı bitirmesini bekle
            last = mtimes(nb_paths)
            try:
                build(nb_paths, out_path, title)
            except Exception as e:           # bozuk/yarım dosyada sunumu çökertme
                print("uyarı: notebook okunamadı, bir sonraki kayıtta tekrar denenecek (%s)" % e, flush=True)


def main():
    ap = argparse.ArgumentParser(description="Notebook grafiklerini canlı HTML sunuma çevirir")
    ap.add_argument("--nb", nargs="+", default=None,
                    help="notebook dosya(ları); verilmezse fraud_detection_eda.ipynb ve fraud_detection_models.ipynb (varsa) kullanılır")
    ap.add_argument("--out", default="sunum.html", help="üretilecek statik HTML dosyası")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--title", default="Kredi Kartı Dolandırıcılık Tespiti — Sunum")
    ap.add_argument("--once", action="store_true", help="sadece HTML üret, sunucu başlatma")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()

    if a.nb is None:
        a.nb = [p for p in ("fraud_detection_eda.ipynb", "fraud_detection_models.ipynb") if os.path.exists(p)]
    a.nb = [p for p in a.nb if os.path.exists(p)]
    if not a.nb:
        sys.exit("Notebook bulunamadı. sunum.py'yi notebook ile aynı klasörde çalıştır veya --nb ile yolu ver.")
    print("Kullanılan notebook'lar:", ", ".join(a.nb))

    build(a.nb, a.out, a.title)
    if a.once:
        print("Hazır: %s  (çift tıklayıp tarayıcıda açabilirsin)" % a.out)
        return

    server = None
    for port in range(a.port, a.port + 10):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
            break
        except OSError:
            continue
    if server is None:
        sys.exit("Boş port bulunamadı. --port ile başka bir port dene.")

    threading.Thread(target=watch, args=(a.nb, a.out, a.title), daemon=True).start()
    url = "http://localhost:%d" % server.server_address[1]
    print("\nSunum hazır -> %s" % url)
    print("Notebook'ta hücreleri çalıştırıp Ctrl+S ile kaydettiğinde sayfa kendiliğinden yenilenir.")
    print("Durdurmak için Ctrl+C\n", flush=True)
    if not a.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nKapatıldı.")


if __name__ == "__main__":
    main()
