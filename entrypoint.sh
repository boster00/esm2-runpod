#!/usr/bin/env bash
# RUN_MODE=proteome -> baked proteome precompute script; else the serverless handler.
set -e
if [ "$RUN_MODE" = "proteome" ]; then
  exec python -u run_proteome_plm.py
else
  exec python -u handler.py
fi
