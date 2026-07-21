#!/usr/bin/env python3
"""
run_proteome_plm.py — Proteome-wide PLM per-residue epitope-quality precompute.

Self-contained GPU pod script:
  1. Train logistic-regression classifier on IEDB linear B-cell epitope peptides
     (per-residue ESM-2 last-hidden-state features, dim=640 for 650M model).
  2. Fetch all canonical human / mouse / rat protein sequences from Supabase.
  3. Dedup: within each species, keep the LONGER sequence per gene symbol.
  4. Score each protein: per-residue ESM-2 embedding → LR prob → 0-9 digit.
  5. Write compressed JSONL to local file, then upload to Supabase Storage.
  6. Write a manifest + status record so the local orchestrator can detect completion.

Never persists raw embeddings — only the 0-9 digit strings are written.

Run on RunPod (or any CUDA host):
  pip install fair-esm scikit-learn numpy requests
  python run_proteome_plm.py

Required env vars (or edit the CONFIG block below):
  SUPABASE_URL, SUPABASE_SERVICE_KEY
  SUPABASE_BUCKET  (default: plm-scores)
"""

import csv
import gzip
import io
import json
import os
import sys
import time
import urllib.request
import urllib.error
from collections import defaultdict
from datetime import datetime

import numpy as np

# ─── CONFIG ───────────────────────────────────────────────────────────────────

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")
SUPABASE_BUCKET = os.environ.get("SUPABASE_BUCKET", "plm-scores")

IEDB_URL = ("https://raw.githubusercontent.com/"
            "boster00/esm2-runpod/master/data.csv")

MODEL_NAME = os.environ.get("ESM_MODEL", "facebook/esm2_t33_650M_UR50D")

CHUNK_SIZE = 1000      # max AA per ESM-2 forward pass
CHUNK_OVERLAP = 100    # AA of overlap for long proteins

TARGET_SPECIES = {
    "human": "human",
    "Human": "human",
    "Mouse": "mouse",
    "Rat":   "rat",
}
PAGE = 1000   # Supabase page size

LOCAL_OUT = "/tmp/plm_scores.jsonl.gz"

LOG = lambda *a: print(*a, flush=True)


# ─── Supabase helpers ──────────────────────────────────────────────────────────

def sb_request(method, path, body=None, extra_headers=None):
    url = f"{SUPABASE_URL}/{path}"
    data = json.dumps(body).encode() if body else None
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        resp = urllib.request.urlopen(req, timeout=120)
        return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def fetch_sequences(species_code):
    rows = []
    offset = 0
    while True:
        path = (
            f"rest/v1/atgd_gene_info"
            f"?select=id,symbol,sequence"
            f"&species_code=eq.{species_code}"
            f"&sequence=not.is.null"
            f"&order=id"
            f"&limit={PAGE}&offset={offset}"
        )
        req = urllib.request.Request(
            f"{SUPABASE_URL}/{path}",
            headers={
                "apikey": SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}",
                "Prefer": "count=exact",
            }
        )
        resp = urllib.request.urlopen(req, timeout=120)
        batch = json.loads(resp.read())
        cr = resp.headers.get("Content-Range", "")
        total = int(cr.split("/")[1]) if "/" in cr else len(batch)
        rows.extend(batch)
        LOG(f"  {species_code}: {len(rows)}/{total}")
        if len(rows) >= total or not batch:
            break
        offset += PAGE
    return rows


def dedup(rows, species_label):
    best = {}
    for r in rows:
        sym = (r.get("symbol") or "").strip()
        seq = (r.get("sequence") or "").strip()
        if not sym or not seq:
            continue
        if sym not in best or len(seq) > len(best[sym]["sequence"]):
            best[sym] = {
                "gene_id": r["id"],
                "symbol": sym,
                "species": species_label,
                "sequence": seq,
            }
    return list(best.values())


def upload_to_supabase(local_path, remote_name):
    """Upload a local file to Supabase Storage."""
    with open(local_path, "rb") as f:
        data = f.read()
    url = f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}/{remote_name}"
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/octet-stream",
            "x-upsert": "true",
        }
    )
    resp = urllib.request.urlopen(req, timeout=120)
    return json.loads(resp.read())


def upsert_status(record):
    """Upsert a row into plm_precompute_status (create if needed)."""
    # Try to create the table if it doesn't exist via a simple insert
    # (just write to a well-known Supabase key-value store or bucket)
    # We'll store status as a JSON file in the same bucket
    data = json.dumps(record).encode()
    url = f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}/status.json"
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json",
            "x-upsert": "true",
        }
    )
    try:
        urllib.request.urlopen(req, timeout=30)
    except Exception as e:
        LOG(f"  status write warn: {e}")


# ─── ESM-2 + classifier ───────────────────────────────────────────────────────

def setup_esm():
    import torch
    import esm as esm_lib

    LOG("Loading ESM-2 650M …")
    model, alphabet = esm_lib.pretrained.esm2_t33_650M_UR50D()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    LOG(f"  device: {device}")
    model = model.eval().to(device)
    bc = alphabet.get_batch_converter()
    return model, alphabet, bc, device


def embed_per_residue(seq, model, alphabet, bc, device):
    """Embed a single sequence. Returns np [L, 1280] (650M dim=1280)."""
    import torch

    seq = seq.upper().strip()[:1022]
    data = [("p", seq)]
    _, _, toks = bc(data)
    toks = toks.to(device)
    with torch.no_grad():
        reps = model(toks, repr_layers=[33])["representations"][33]
    # reps shape: [1, L_with_special, 1280]
    # strip BOS (index 0) and EOS (last non-padding index)
    n_real = (toks[0] != alphabet.padding_idx).sum().item() - 2
    return reps[0, 1:n_real + 1].cpu().numpy()   # [n_real, H]


def train_classifier(model, alphabet, bc, device):
    from sklearn.linear_model import LogisticRegression

    LOG("Fetching IEDB training data …")
    raw = urllib.request.urlopen(IEDB_URL, timeout=60).read().decode()
    seqs, ys = [], []
    for row in csv.DictReader(io.StringIO(raw)):
        seqs.append(row["seq"].upper())
        ys.append(int(row["y"]))
    LOG(f"  {len(seqs)} training peptides "
        f"({sum(ys)} epitope / {len(ys)-sum(ys)} non-epitope)")

    all_embs, all_labels = [], []
    for i, (seq, y) in enumerate(zip(seqs, ys)):
        emb = embed_per_residue(seq, model, alphabet, bc, device)
        for vec in emb:
            all_embs.append(vec)
            all_labels.append(y)
        if i % 200 == 0:
            LOG(f"  embedded {i}/{len(seqs)} training peptides …")

    X = np.array(all_embs, dtype=np.float32)
    y_arr = np.array(all_labels, dtype=np.int8)
    LOG(f"Training LR on {len(y_arr):,} residues …")
    clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs",
                             n_jobs=-1).fit(X, y_arr)
    LOG("  Classifier ready.")
    return clf


def score_protein(seq, clf, model, alphabet, bc, device):
    """Return 0-9 digit string, length == len(seq)."""
    seq = seq.upper().strip()
    L = len(seq)

    if L <= CHUNK_SIZE:
        emb = embed_per_residue(seq, model, alphabet, bc, device)
        n = min(len(emb), L)
        probs = clf.predict_proba(emb[:n])[:, 1]
        if n < L:
            pad = float(np.median(probs)) if len(probs) else 0.5
            probs = np.concatenate([probs, np.full(L - n, pad)])
    else:
        probs_sum = np.zeros(L, dtype=np.float64)
        probs_cnt = np.zeros(L, dtype=np.float64)
        start = 0
        while start < L:
            end = min(start + CHUNK_SIZE, L)
            emb = embed_per_residue(seq[start:end], model, alphabet, bc, device)
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


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    LOG(f"=== PLM Proteome Precompute — {datetime.utcnow().isoformat()}Z ===")

    # Step 1: install deps
    LOG("Installing dependencies …")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                   "fair-esm", "scikit-learn", "numpy"], check=True)

    # Step 2: fetch + dedup
    LOG("\n=== Fetching sequences from Supabase ===")
    raw_by_species = defaultdict(list)
    for sc, label in TARGET_SPECIES.items():
        rows = fetch_sequences(sc)
        raw_by_species[label].extend(rows)

    LOG("\n=== Deduplication ===")
    canonical = {}
    stats_raw = {}
    for label, rows in raw_by_species.items():
        stats_raw[label] = len(rows)
        canon = dedup(rows, label)
        canonical[label] = canon
        LOG(f"  {label}: {len(rows)} raw -> {len(canon)} canonical")

    all_proteins = []
    for label in ("human", "mouse", "rat"):
        all_proteins.extend(canonical.get(label, []))
    total_proteins = len(all_proteins)
    total_residues = sum(len(p["sequence"]) for p in all_proteins)
    LOG(f"\nTotal canonical proteins: {total_proteins:,}")
    LOG(f"Total residues:           {total_residues:,}")

    # Step 3: load ESM-2 + train classifier
    model, alphabet, bc, device = setup_esm()
    clf = train_classifier(model, alphabet, bc, device)
    t_clf = time.time() - t0
    LOG(f"Setup complete in {t_clf:.0f}s")

    # Step 4: score all proteins
    LOG("\n=== Scoring proteins ===")
    scored = 0
    with gzip.open(LOCAL_OUT, "wt", encoding="utf-8") as fh:
        for prot in all_proteins:
            seq = prot["sequence"]
            score_str = score_protein(seq, clf, model, alphabet, bc, device)
            rec = {
                "gene_id": prot["gene_id"],
                "symbol":  prot["symbol"],
                "species": prot["species"],
                "length":  len(seq),
                "scores":  score_str,
            }
            fh.write(json.dumps(rec) + "\n")
            scored += 1
            if scored % 500 == 0:
                elapsed = time.time() - t0
                rate = scored / elapsed
                eta = (total_proteins - scored) / rate if rate else 0
                LOG(f"  {scored}/{total_proteins} proteins "
                    f"({elapsed:.0f}s elapsed, ETA {eta:.0f}s)")

    elapsed_total = time.time() - t0
    LOG(f"\nAll {scored} proteins scored in {elapsed_total:.0f}s "
        f"({elapsed_total/3600:.2f}h)")

    # Step 5: upload to Supabase Storage
    LOG("\n=== Uploading to Supabase Storage ===")
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    remote_name = f"scores_{ts}.jsonl.gz"
    try:
        upload_to_supabase(LOCAL_OUT, remote_name)
        LOG(f"  Uploaded: {SUPABASE_BUCKET}/{remote_name}")
    except Exception as e:
        LOG(f"  Upload error: {e}")
        remote_name = None

    # Step 6: manifest + status
    manifest = {
        "schema_version": "1.0",
        "date": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": MODEL_NAME,
        "classifier": {
            "type": "LogisticRegression",
            "training_data": "IEDB linear B-cell epitope peptides (data.csv, 2000 records)",
            "training_url": IEDB_URL,
            "features": "per-residue ESM-2 last-hidden-state (fair-esm layer 33), dim=1280",
            "label": "1 for each residue in epitope peptide (y=1), 0 for non-epitope",
            "sklearn_params": "LogisticRegression(C=1.0, solver=lbfgs, max_iter=1000)",
        },
        "dedup_rule": (
            "Within each species, for duplicate gene symbols keep the LONGER sequence. "
            "human + Human species_code variants merged before dedup."
        ),
        "chunking": {
            "chunk_size_aa": CHUNK_SIZE,
            "overlap_aa": CHUNK_OVERLAP,
            "merge": "average probabilities at overlap positions",
        },
        "score_encoding": "digit string 0-9 per residue; 0=lowest, 9=highest epitope quality",
        "output_schema": {
            "gene_id": "integer PK from atgd_gene_info",
            "symbol": "gene symbol",
            "species": "human | mouse | rat",
            "length": "protein length AA",
            "scores": "digit string len==length",
        },
        "counts": {
            "human_raw": stats_raw.get("human", 0),
            "mouse_raw": stats_raw.get("mouse", 0),
            "rat_raw":   stats_raw.get("rat", 0),
            "human_canonical": len(canonical.get("human", [])),
            "mouse_canonical": len(canonical.get("mouse", [])),
            "rat_canonical":   len(canonical.get("rat", [])),
            "total_proteins":  total_proteins,
            "total_residues":  total_residues,
            "scored":          scored,
        },
        "timing": {
            "total_seconds": round(elapsed_total),
            "hours":         round(elapsed_total / 3600, 2),
        },
        "storage": {
            "bucket": SUPABASE_BUCKET,
            "file":   remote_name,
        },
        "runpod": {
            "device": device,
        },
    }

    manifest_local = "/tmp/plm_manifest.json"
    with open(manifest_local, "w") as f:
        json.dump(manifest, f, indent=2)
    LOG(f"Manifest written to {manifest_local}")

    try:
        upload_to_supabase(manifest_local, "manifest.json")
        LOG("  Manifest uploaded.")
    except Exception as e:
        LOG(f"  Manifest upload error: {e}")

    status = {"status": "complete", "scored": scored, "remote_name": remote_name,
              "date": manifest["date"]}
    upsert_status(status)
    LOG("\nDONE.", json.dumps(status))


if __name__ == "__main__":
    main()
