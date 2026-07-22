# Engineering Decisions — esm2-runpod

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
