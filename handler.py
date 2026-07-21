"""
ESM-2 RunPod worker v2: mean-pooled embeddings (backward compat) + per-residue
epitope-quality scoring for proteome-scale precompute.

Actions
-------
ping               — health / warm-up check
embed              — original mean-pooled embeddings per sequence (v1 compat)
score_batch        — per-residue 0-9 epitope-quality strings for a list of proteins
                     Input:  { "entries": [{"gene_id":..,"symbol":..,"species":..,"sequence":..}, ...] }
                     Output: { "results": [{"gene_id":..,"symbol":..,"species":..,"length":..,"scores":"034.."},..] }

Classifier: logistic regression trained on per-residue ESM-2 last-hidden-state
embeddings from 2000 IEDB linear B-cell epitope peptides (data.csv).  Each
residue in a positive peptide → label 1; each in a negative peptide → label 0.
Classifier is trained once on the first warm worker and cached globally.

Chunking: proteins > CHUNK_SIZE (1000 AA) are sliced with CHUNK_OVERLAP (100 AA)
overlap; per-residue probabilities at overlap positions are averaged.

ESM-2 tokenizer adds CLS at position 0 and EOS after the last real token;
we strip both so the returned embedding tensor is exactly [len(seq), hidden_dim].
"""
import os
import csv
import io
import json
import urllib.request

import runpod
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression

MODEL = os.environ.get("ESM_MODEL", "facebook/esm2_t33_650M_UR50D")
IEDB_URL = "https://raw.githubusercontent.com/boster00/esm2-runpod/master/data.csv"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

_DEV = "cuda" if torch.cuda.is_available() else "cpu"
_tok = None
_model = None
_clf = None          # cached LR classifier; trained once per warm worker


# ─── ESM-2 loading ────────────────────────────────────────────────────────────

def _load_esm():
    global _tok, _model
    if _model is None:
        _tok = AutoTokenizer.from_pretrained(MODEL)
        _model = AutoModel.from_pretrained(MODEL).to(_DEV).eval()
    return _tok, _model


# ─── Embedding helpers ────────────────────────────────────────────────────────

def _embed_mean_batch(seqs, batch=8):
    """Mean-pooled last-hidden-state per sequence (v1 behaviour)."""
    tok, model = _load_esm()
    out = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch):
            chunk = seqs[i:i + batch]
            enc = tok(chunk, return_tensors="pt", padding=True,
                      truncation=True, max_length=1024).to(_DEV)
            rep = model(**enc).last_hidden_state          # [B, L, H]
            mask = enc["attention_mask"].unsqueeze(-1).float()
            mean = (rep * mask).sum(1) / mask.sum(1).clamp(min=1)
            out.extend(mean.cpu().tolist())
    return out


def _embed_per_residue(seq):
    """Per-residue last-hidden-state for ONE sequence.  Returns np.ndarray [L, H].

    ESM-2 tokenizer layout: [CLS] tok_0 tok_1 … tok_{L-1} [EOS]
    We strip CLS (index 0) and EOS (the last non-padding token) so the result
    aligns 1-to-1 with seq characters.  Sequences are truncated at 1022 AA
    (1024 tokens minus the two special tokens).
    """
    tok, model = _load_esm()
    seq = seq.upper().strip()[:1022]          # hard cap matches tokenizer truncation
    enc = tok([seq], return_tensors="pt",
              truncation=True, max_length=1024).to(_DEV)
    with torch.no_grad():
        rep = model(**enc).last_hidden_state  # [1, L_tok, H]
    # real residue count = attention tokens - CLS - EOS
    n_real = int(enc["attention_mask"][0].sum().item()) - 2
    return rep[0, 1:n_real + 1, :].cpu().numpy()   # [n_real, H]


# ─── Classifier ───────────────────────────────────────────────────────────────

def _train_clf():
    """Build + cache a logistic-regression epitope classifier.

    Training data: 2 000 IEDB linear B-cell epitope peptides from data.csv.
    Features: per-residue ESM-2 last-hidden-state (640 d for 650 M model).
    Label:    1 for every residue in an epitope peptide, 0 for a non-epitope.

    Returns the fitted sklearn LogisticRegression.
    """
    global _clf
    if _clf is not None:
        return _clf

    raw = urllib.request.urlopen(IEDB_URL, timeout=60).read().decode()
    seqs, ys = [], []
    for row in csv.DictReader(io.StringIO(raw)):
        seqs.append(row["seq"].upper())
        ys.append(int(row["y"]))

    all_embs, all_labels = [], []
    for seq, y in zip(seqs, ys):
        emb = _embed_per_residue(seq)           # [L, H]
        for vec in emb:
            all_embs.append(vec)
            all_labels.append(y)

    X = np.array(all_embs, dtype=np.float32)
    y_arr = np.array(all_labels, dtype=np.int8)
    _clf = LogisticRegression(
        max_iter=1000, C=1.0, solver="lbfgs", n_jobs=-1
    ).fit(X, y_arr)
    print(f"[classifier] trained on {len(all_labels)} residues "
          f"({sum(all_labels)} epitope, {len(all_labels)-sum(all_labels)} non-epitope)",
          flush=True)
    return _clf


# ─── Protein scoring ──────────────────────────────────────────────────────────

def _score_protein(seq):
    """Return a digit string '0'-'9', one char per residue (len == len(seq)).

    Long proteins (> CHUNK_SIZE) are embedded in overlapping chunks and the
    per-residue probabilities at overlap positions are averaged before
    quantisation.
    """
    clf = _train_clf()
    seq = seq.upper().strip()
    L = len(seq)

    if L <= CHUNK_SIZE:
        emb = _embed_per_residue(seq)       # may be shorter if somehow truncated
        n = min(len(emb), L)
        probs = clf.predict_proba(emb[:n])[:, 1]
        if n < L:                           # pad rare truncation with median
            pad = float(np.median(probs)) if len(probs) else 0.5
            probs = np.concatenate([probs, np.full(L - n, pad)])
    else:
        probs_sum = np.zeros(L, dtype=np.float64)
        probs_cnt = np.zeros(L, dtype=np.float64)
        start = 0
        while start < L:
            end = min(start + CHUNK_SIZE, L)
            emb = _embed_per_residue(seq[start:end])
            n = len(emb)
            p = clf.predict_proba(emb)[:, 1]
            probs_sum[start:start + n] += p
            probs_cnt[start:start + n] += 1
            if end >= L:
                break
            start = end - CHUNK_OVERLAP
        probs = probs_sum / np.maximum(probs_cnt, 1)

    scores = np.clip((probs * 10).astype(int), 0, 9)
    return "".join(map(str, scores.tolist()))


# ─── RunPod handler ───────────────────────────────────────────────────────────

def handler(job):
    ji = job.get("input") if isinstance(job.get("input"), dict) else {}
    action = ji.get("action", "embed")

    # ── ping ──────────────────────────────────────────────────────────────────
    if action == "ping":
        _load_esm()
        return {"message": "esm2-v2 ok", "device": _DEV, "model": MODEL,
                "clf_cached": _clf is not None}

    # ── embed (v1 compat) ─────────────────────────────────────────────────────
    if action == "embed":
        seqs = [str(s).upper().strip() for s in (ji.get("sequences") or [])]
        if not seqs:
            return {"error": "provide input.sequences: [str, ...]"}
        emb = _embed_mean_batch(seqs, int(ji.get("batch", 8)))
        return {"model": MODEL, "n": len(emb),
                "dim": len(emb[0]) if emb else 0, "embeddings": emb}

    # ── score_batch ───────────────────────────────────────────────────────────
    if action == "score_batch":
        entries = ji.get("entries") or []
        if not entries:
            return {"error": "provide input.entries: [{gene_id, symbol, species, sequence}, ...]"}
        results = []
        for e in entries:
            seq = (e.get("sequence") or "").upper().strip()
            if not seq:
                continue
            score_str = _score_protein(seq)
            results.append({
                "gene_id": e.get("gene_id"),
                "symbol": e.get("symbol"),
                "species": e.get("species"),
                "length": len(seq),
                "scores": score_str,
            })
        return {
            "model": MODEL,
            "clf_trained": _clf is not None,
            "count": len(results),
            "results": results,
        }

    return {"error": f"unknown action: {action!r}"}


runpod.serverless.start({"handler": handler})
