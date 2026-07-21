"""
Self-contained study job. Uses fair-esm (loads ESM-2 directly on torch, bypassing
transformers' backend detection). Embeds 2000 IEDB peptides, 5-fold CV logistic on
ESM-2 embeddings vs classical dipeptide composition, POSTs RESULT (or error) to $WEBHOOK.
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
    print("RESULT", json.dumps(obj)[:400], flush=True)

try:
    post({"stage": "started"})
    import numpy as np, torch, esm
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    raw = urllib.request.urlopen(
        "https://raw.githubusercontent.com/boster00/esm2-runpod/master/data.csv", timeout=60).read().decode()
    seqs, ys = [], []
    for row in csv.DictReader(io.StringIO(raw)):
        seqs.append(row["seq"].upper()); ys.append(int(row["y"]))
    ys = np.array(ys)
    post({"stage": "loaded", "n": len(seqs)})

    AA = "ACDEFGHIKLMNPQRSTVWY"; idx = {a+b: i for i, a in enumerate(AA) for b in AA}
    def dpc(s):
        v = np.zeros(400)
        for i in range(len(s)-1):
            k = idx.get(s[i:i+2])
            if k is not None: v[k] += 1
        return v/max(1, len(s)-1)
    Xc = np.vstack([dpc(s) for s in seqs])

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.eval().to(dev)
    bc = alphabet.get_batch_converter()
    def embed(chunk):
        data = [(str(i), s[:1022]) for i, s in enumerate(chunk)]
        _, _, toks = bc(data); toks = toks.to(dev)
        with torch.no_grad():
            reps = model(toks, repr_layers=[33])["representations"][33]
        lens = (toks != alphabet.padding_idx).sum(1)
        return np.vstack([reps[i, 1:lens[i]-1].mean(0).cpu().numpy() for i in range(len(chunk))])
    Xe = np.vstack([embed(seqs[i:i+16]) for i in range(0, len(seqs), 16)])
    post({"stage": "embedded", "shape": list(Xe.shape)})

    def cv_auc(X):
        skf = StratifiedKFold(5, shuffle=True, random_state=0); a = []
        for tr, te in skf.split(X, ys):
            clf = LogisticRegression(max_iter=3000, C=1.0).fit(X[tr], ys[tr])
            a.append(roc_auc_score(ys[te], clf.predict_proba(X[te])[:, 1]))
        return float(np.mean(a)), float(np.std(a))

    ca, cs = cv_auc(Xc); ea, es = cv_auc(Xe)
    post({"stage": "done", "device": dev, "n": len(seqs), "pos": int(ys.sum()),
          "classical_dpc_auroc": round(ca, 4), "esm2_auroc": round(ea, 4), "delta": round(ea-ca, 4)})
except Exception:
    post({"stage": "error", "traceback": traceback.format_exc()[-1500:]})
