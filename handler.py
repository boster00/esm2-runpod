"""
RunPod serverless worker: ESM-2 protein embeddings.

Input  : { "input": { "sequences": ["SEQ1", "SEQ2", ...], "batch": 8, "action": "ping"? } }
Output : { "model": ..., "n": N, "dim": D, "embeddings": [[...D...], ...N...] }

Mean-pooled last-hidden-state per sequence (masked over real tokens). GPU if available.
"""
import os
import runpod
import torch
from transformers import AutoTokenizer, AutoModel

MODEL = os.environ.get("ESM_MODEL", "facebook/esm2_t33_650M_UR50D")
_DEV = "cuda" if torch.cuda.is_available() else "cpu"
_tok = None
_model = None


def _load():
    global _tok, _model
    if _model is None:
        _tok = AutoTokenizer.from_pretrained(MODEL)
        _model = AutoModel.from_pretrained(MODEL).to(_DEV).eval()
    return _tok, _model


def embed(seqs, batch=8):
    tok, model = _load()
    out = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch):
            chunk = seqs[i:i + batch]
            enc = tok(chunk, return_tensors="pt", padding=True,
                      truncation=True, max_length=1024).to(_DEV)
            rep = model(**enc).last_hidden_state          # [B, L, H]
            mask = enc["attention_mask"].unsqueeze(-1).float()
            mean = (rep * mask).sum(1) / mask.sum(1).clamp(min=1)  # masked mean-pool
            out.extend(mean.cpu().tolist())
    return out


def handler(job):
    ji = job.get("input") if isinstance(job.get("input"), dict) else {}
    if ji.get("action") == "ping":
        return {"message": "esm2 ok", "device": _DEV, "model": MODEL}
    seqs = ji.get("sequences") or []
    if not seqs:
        return {"error": "provide input.sequences: [str, ...]"}
    seqs = [str(s).upper().strip() for s in seqs]
    emb = embed(seqs, int(ji.get("batch", 8)))
    return {"model": MODEL, "n": len(emb), "dim": (len(emb[0]) if emb else 0), "embeddings": emb}


runpod.serverless.start({"handler": handler})
