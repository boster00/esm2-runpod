"""
Self-contained study job: does ESM-2 embedding beat classical composition on the
same peptides? Reads data.csv (seq,y), embeds with ESM-2, 5-fold CV logistic,
prints AUROC for BOTH ESM-2 embeddings and a classical dipeptide-composition baseline.
"""
import csv, numpy as np, torch
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

seqs, ys = [], []
with open("data.csv") as f:
    for row in csv.DictReader(f):
        seqs.append(row["seq"].upper()); ys.append(int(row["y"]))
ys = np.array(ys)
print(f"loaded {len(seqs)} seqs, {ys.sum()} pos / {(ys==0).sum()} neg", flush=True)

# ---- classical dipeptide composition baseline ----
AA = "ACDEFGHIKLMNPQRSTVWY"; idx = {a+b: i for i, a in enumerate(AA) for b in AA}
def dpc(s):
    v = np.zeros(400)
    for i in range(len(s)-1):
        k = idx.get(s[i:i+2])
        if k is not None: v[k] += 1
    return v/max(1, len(s)-1)
Xc = np.vstack([dpc(s) for s in seqs])

# ---- ESM-2 embeddings ----
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev, flush=True)
tok = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
model = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D").to(dev).eval()
def embed(chunk):
    enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=64).to(dev)
    with torch.no_grad():
        rep = model(**enc).last_hidden_state
    mask = enc["attention_mask"].unsqueeze(-1).float()
    return ((rep*mask).sum(1)/mask.sum(1).clamp(min=1)).cpu().numpy()
Xe = np.vstack([embed(seqs[i:i+16]) for i in range(0, len(seqs), 16)])
print(f"embedded {Xe.shape}", flush=True)

def cv_auc(X):
    skf = StratifiedKFold(5, shuffle=True, random_state=0); a = []
    for tr, te in skf.split(X, ys):
        clf = LogisticRegression(max_iter=3000, C=1.0).fit(X[tr], ys[tr])
        a.append(roc_auc_score(ys[te], clf.predict_proba(X[te])[:, 1]))
    return np.mean(a), np.std(a)

ca, cs = cv_auc(Xc); ea, es = cv_auc(Xe)
print(f"RESULT classical_dpc_AUROC={ca:.4f}+/-{cs:.4f}", flush=True)
print(f"RESULT esm2_AUROC={ea:.4f}+/-{es:.4f}", flush=True)
print(f"RESULT delta={ea-ca:+.4f}", flush=True)
