# Engineering Decisions — esm2-runpod

## 2026-07-22b · plm digit = PER-PROTEIN relative epitope quality, NOT absolute prob×10

**Decision:** `score_protein` maps each protein's per-residue probabilities to 0-9 via
**per-protein PERCENTILE-RANK** (each residue's rank among the protein's residues → 0-9),
not `clip(prob*10)`.

**Rejected alternatives:** (1) `clip(prob*10)` — absolute probability × 10 (the original);
(2) global/cross-protein normalization; (3) **per-protein min-max (2nd–98th pct)** — tried
first, but the within-protein prob distribution is itself right-skewed so it still left
**~82% of residues at digit 9** (measured on a 1,000-protein re-score). Rank normalization
forces an even 0-9 spread by construction regardless of distribution shape, which is the
only thing that reliably de-saturates the track.

**Why:** measured across a 1,600-protein / 909K-residue sample, the raw ESM-2+LR
probability saturates — **92.9% of all residues mapped to digit 9**, giving a nearly-flat
track pinned at max (the user's report: "most scores are 100, not useful"). The classifier
still discriminates (~0.81 AUROC on balanced epitope peptides), so the *relative ordering
within a protein* is informative even though absolute probs cluster high. Global
normalization does NOT fix this — most proteins are genuinely mostly-high, so they'd still
read as saturated; only PER-PROTEIN normalization spreads every protein's line and surfaces
its internal hotspots, which is what the antigen-design "Scoring" histogram needs.

**Consequence if violated:** revert to `prob*10` → the purple PLM line flatlines at ~+100
for ~93% of positions and conveys no per-residue signal. **Also:** because only the clipped
0-9 digit is stored (not the raw prob), any future re-mapping requires a full re-embed/
re-score — the spread cannot be recovered from stored digits.



## 2026-07-22 · run_proteome_plm.py writes a resumable partial + progress.json to storage every 500 proteins

**Decision:** during the scoring loop, every 500 proteins we (a) re-upload the full
cumulative `partial.jsonl.gz` and (b) upload a `progress.json` {scored,total,pct,status}
to the `plm-scores` bucket. On boot the script downloads `partial.jsonl.gz` and skips
already-scored `gene_id`s.

**Rejected alternative:** the original v1 wrote one gzip and uploaded only at the very
end (single final `upload_to_supabase`).

**Why:** RunPod **on-demand pod stdout is NOT accessible via any API** — GraphQL has no
`logs` field and `rest.runpod.io/v1/pods/{id}/logs` returns 400 (web-console websocket
only). With end-only upload, an orchestrator babysitting the run is BLIND for the whole
~40-min run and only learns success/failure at the end — which is exactly how the prior
"compute finished but never persisted" failures went undetected. The storage-side
`progress.json` is the only live progress signal available, and the resumable partial
means a pod death mid-run costs minutes, not the whole run.

**Consequence if violated:** revert to end-only upload → the run is unobservable and
non-resumable again; a silent mid-run failure burns ~$0.30 + 40 min undetected, and any
crash restarts from zero.
