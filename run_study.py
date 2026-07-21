"""
Self-contained study job. Fetches data.csv from raw GitHub, embeds with ESM-2,
5-fold CV logistic on ESM-2 embeddings vs classical dipeptide composition, and
POSTs the RESULT (or any error) to $WEBHOOK so it can be read without pod logs.
"""
import os, csv, io, json, urllib.request, traceback

WEBHOOK = os.environ.get("WEBHOOK", "")
def post(obj):
    try:
        if WEBHOOK:
            urllib.request.urlopen(urllib.request.Request(
                WEBHOOK, data=json.dumps(obj).encode(),
                headers={"Content-Type": "application/json"}), timeout=30)
    except Exception as e:
        print("post failed", e, flush=True)
    print("RESULT", json.dumps(obj), flush=True)

try:
    post({"stage": "started"})
    import numpy as np, torch
    from transformers import AutoTokenizer, AutoModel
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    raw = urllib.request.urlopen(
        "https://raw.githubusercontent.com/boster00/esm2-runpod/master/data.csv", timeout=60).read().decode()
    seqs, ys = [], []
    for row in csv.DictReader(io.StringIO(raw)):
        seqs.append(row["seq"].upper()); ys.append(int(row["y"]))
    ys = np.array(ys)

    AA = "ACDEFGHIKLMNPQRSTVWY"; idx = {a+b: i for i, a in enumerate(AA) for b in AA}
    def dpc(s):
        v = np.zeros(400)
        for i in range(len(s)-1):
            k = idx.get(s[i:i+2])
            if k is not None: v[k] += 1
        return v/max(1, len(s)-1)
    Xc = np.vstack([dpc(s) for s in seqs])

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    model = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D").to(dev).eval()
    def embed(chunk):
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=64).to(dev)
        with torch.no_grad():
            rep = model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        return ((rep*mask).sum(1)/mask.sum(1).clamp(min=1)).cpu().numpy()
    Xe = np.vstack([embed(seqs[i:i+16]) for i in range(0, len(seqs), 16)])

    def cv_auc(X):
        skf = StratifiedKFold(5, shuffle=True, random_state=0); a = []
        for tr, te in skf.split(X, ys):
            clf = LogisticRegression(max_iter=3000, C=1.0).fit(X[tr], ys[tr])
            a.append(roc_auc_score(ys[te], clf.predict_proba(X[te])[:, 1]))
        return float(np.mean(a)), float(np.std(a))

    ca, cs = cv_auc(Xc); ea, es = cv_auc(Xe)
    post({"stage": "done", "device": dev, "n": len(seqs), "pos": int(ys.sum()),
          "classical_dpc_auroc": round(ca, 4), "classical_std": round(cs, 4),
          "esm2_auroc": round(ea, 4), "esm2_std": round(es, 4), "delta": round(ea-ca, 4)})
except Exception:
    post({"stage": "error", "traceback": traceback.format_exc()[-1500:]})
