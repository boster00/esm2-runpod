"""
ESM-2 RunPod worker v3: fair-esm per-residue epitope-quality scoring.

Uses fair-esm (same as v4 on-pod script, proven to work) instead of
HuggingFace transformers to avoid torch-detection conflicts in conda env.

Actions
-------
ping          — health / warm-up check
embed         — mean-pooled embeddings per sequence (v1 compat)
score_batch   — per-residue 0-9 epitope-quality strings for a list of proteins
               Input:  { "entries": [{"gene_id","symbol","species","sequence"}, ...] }
               Output: { "results": [{"gene_id","symbol","species","length","scores":"034.."}, ...] }

Classifier: logistic regression trained on ESM-2 650M last-hidden-state embeddings
from IEDB linear B-cell epitope peptides (data.csv bundled in image).
Trained once on first job and cached in process memory.
"""
import os
import csv
import io
import json

import runpod
import torch
import numpy as np

MAX_LEN      = 1022
CHUNK_OVERLAP = 100
DATA_CSV     = os.path.join(os.path.dirname(__file__), "data.csv")
IEDB_URL     = "https://raw.githubusercontent.com/boster00/esm2-runpod/master/data.csv"

_DEV   = "cuda" if torch.cuda.is_available() else "cpu"
_model = None
_alphabet = None
_bc    = None   # batch converter
_clf   = None   # cached LR classifier


# ─── ESM-2 loading ────────────────────────────────────────────────────────────

def _load_esm():
    global _model, _alphabet, _bc
    if _model is None:
        import esm
        print(f"[ESM-2] Loading esm2_t33_650M_UR50D on {_DEV}...", flush=True)
        _model, _alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        _model = _model.to(_DEV).eval()
        _bc = _alphabet.get_batch_converter()
        print(f"[ESM-2] Loaded. hidden_size=1280", flush=True)
    return _model, _bc


# ─── Embedding helpers ────────────────────────────────────────────────────────

def _embed_per_residue(seq):
    """Per-residue last-hidden-state for ONE sequence. Returns np.ndarray [L, 1280]."""
    model, bc = _load_esm()
    seq = seq.upper().strip()[:MAX_LEN]
    _, _, tokens = bc([("seq", seq)])
    with torch.no_grad():
        out = model(tokens.to(_DEV), repr_layers=[33], return_contacts=False)
    # layer 33 output: [1, L_tok, 1280]; strip CLS (0) and EOS (last)
    rep = out["representations"][33][0, 1:len(seq) + 1].cpu().float().numpy()
    return rep  # [len(seq), 1280]


def _embed_mean_batch(seqs, batch_size=8):
    """Mean-pooled embeddings per sequence (v1 compat). Returns list of 1280-d vectors."""
    model, bc = _load_esm()
    out = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            chunk = seqs[i:i + batch_size]
            data = [(f"s{j}", s.upper().strip()[:MAX_LEN]) for j, s in enumerate(chunk)]
            _, _, tokens = bc(data)
            rep = model(tokens.to(_DEV), repr_layers=[33])["representations"][33]
            for k, (_, seq) in enumerate(data):
                L = len(seq)
                mean = rep[k, 1:L + 1].mean(0).cpu().float().numpy()
                out.append(mean.tolist())
    return out


# ─── Classifier ───────────────────────────────────────────────────────────────

def _load_iedb_csv():
    """Load IEDB training data from bundled data.csv or GitHub fallback."""
    if os.path.exists(DATA_CSV):
        with open(DATA_CSV, encoding="utf-8") as f:
            raw = f.read()
    else:
        import urllib.request
        print(f"[IEDB] data.csv not found locally; downloading from GitHub...", flush=True)
        raw = urllib.request.urlopen(IEDB_URL, timeout=60).read().decode()
    seqs, ys = [], []
    for row in csv.DictReader(io.StringIO(raw)):
        seqs.append(row["seq"].upper())
        ys.append(int(row["y"]))
    return seqs, ys


def _train_clf():
    global _clf
    if _clf is not None:
        return _clf
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    seqs, ys = _load_iedb_csv()
    print(f"[Probe] Training on {len(seqs)} IEDB peptides...", flush=True)

    all_embs, all_labels = [], []
    for seq, y in zip(seqs, ys):
        emb = _embed_per_residue(seq)
        for vec in emb:
            all_embs.append(vec)
            all_labels.append(y)

    X = np.array(all_embs, dtype=np.float32)
    y_arr = np.array(all_labels, dtype=np.int8)
    _clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", n_jobs=-1).fit(X, y_arr)
    auc = roc_auc_score(y_arr, _clf.predict_proba(X)[:, 1])
    print(f"[Probe] AUC={auc:.4f} on {len(all_labels)} residues", flush=True)
    return _clf


# ─── Protein scoring ──────────────────────────────────────────────────────────

def _score_protein(seq):
    """Return digit string '0'-'9', one char per residue."""
    clf = _train_clf()
    seq = seq.upper().strip()
    L = len(seq)

    if L <= MAX_LEN:
        emb = _embed_per_residue(seq)
        n = min(len(emb), L)
        probs = clf.predict_proba(emb[:n])[:, 1]
        if n < L:
            pad = float(np.median(probs)) if len(probs) else 0.5
            probs = np.concatenate([probs, np.full(L - n, pad)])
    else:
        probs_sum = np.zeros(L, dtype=np.float64)
        probs_cnt = np.zeros(L, dtype=np.float64)
        step = MAX_LEN - CHUNK_OVERLAP
        start = 0
        while start < L:
            end = min(start + MAX_LEN, L)
            emb = _embed_per_residue(seq[start:end])
            n = len(emb)
            p = clf.predict_proba(emb)[:, 1]
            probs_sum[start:start + n] += p
            probs_cnt[start:start + n] += 1
            if end >= L:
                break
            start += step
        probs = probs_sum / np.maximum(probs_cnt, 1)

    scores = np.clip((probs * 10).astype(int), 0, 9)
    return "".join(map(str, scores.tolist()))


# ─── RunPod handler ───────────────────────────────────────────────────────────

def handler(job):
    ji = job.get("input") if isinstance(job.get("input"), dict) else {}
    action = ji.get("action", "embed")

    if action == "ping":
        _load_esm()
        return {"message": "esm2-v3 ok (fair-esm)", "device": _DEV,
                "clf_cached": _clf is not None}

    if action == "embed":
        seqs = [str(s).upper().strip() for s in (ji.get("sequences") or [])]
        if not seqs:
            return {"error": "provide input.sequences: [str, ...]"}
        emb = _embed_mean_batch(seqs, int(ji.get("batch", 8)))
        return {"n": len(emb), "dim": len(emb[0]) if emb else 0, "embeddings": emb}

    if action == "score_batch":
        entries = ji.get("entries") or []
        if not entries:
            return {"error": "provide input.entries: [{gene_id, symbol, species, sequence}, ...]"}
        results = []
        for e in entries:
            seq = (e.get("sequence") or "").upper().strip()
            if not seq:
                continue
            try:
                score_str = _score_protein(seq)
            except Exception as ex:
                print(f"[score_batch] ERROR gene_id={e.get('gene_id')}: {ex}", flush=True)
                continue
            results.append({
                "gene_id": e.get("gene_id"),
                "symbol": e.get("symbol"),
                "species": e.get("species"),
                "length": len(seq),
                "scores": score_str,
            })
        return {
            "clf_trained": _clf is not None,
            "count": len(results),
            "results": results,
        }

    return {"error": f"unknown action: {action!r}"}


runpod.serverless.start({"handler": handler})
