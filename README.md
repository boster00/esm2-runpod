# esm2-runpod

RunPod serverless worker that returns **ESM-2 mean-pooled embeddings** for a list of protein/peptide sequences. Built for the BosterBio antigen-design study (the "PLM embeddings ~+0.10 AUROC" lever).

## Deploy (RunPod build-from-GitHub)
1. RunPod Console → Serverless → New Endpoint → **Import Git Repository** → this repo.
2. Dockerfile path: `Dockerfile` (root). GPU: any 16GB+ (e.g. RTX 4090 / A5000).
3. Deploy. First build ~5–10 min.

## Call
```json
POST https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync
Authorization: Bearer <RUNPOD_API_KEY>
{ "input": { "sequences": ["MKT...","DAK..."], "batch": 8 } }
```
Returns `{ model, n, dim, embeddings:[[...D...],...] }`. `action:"ping"` for a health check.

Env: `ESM_MODEL` (default `facebook/esm2_t33_650M_UR50D`).
