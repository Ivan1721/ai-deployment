# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository layout

The actual project lives entirely under `mlops-stack/` — **read
`mlops-stack/CLAUDE.md` for the detailed guide** (setup, commands,
architecture, test suite, CI/CD). This root file only covers what exists
outside that directory.

- `.github/workflows/` — the **real** GitHub Actions workflows (`ci.yml`,
  `cd.yml`, `retrain.yml`, all `runs-on: self-hosted`). The copies under
  `mlops-stack/.github/workflows/` are synced for reference only; GitHub does
  not execute them.
- `paper/` — IEEE conference paper in progress (`conference_101719.tex`,
  IEEEtran class, architecture/flowchart figures). Not tracked by git yet.
- Top-level docs from the P0 security-hardening pass:
  `SECURITY_ASSESSMENT.md`, `CHANGES_P0_SECURITY.md`, `DEPLOYMENT_GUIDE_P0.md`,
  `QUICK_START_P0.md` — read these when touching auth/TLS/networking.
- `DVC_CML_Agents_Setup_Guide.md` — DVC + CML + self-hosted runner setup notes.
- `AGENTS.md` — agent instructions (kept in sync with the CLAUDE.md files).

## What this is

A self-contained MLOps reference stack (Docker Compose, 11 services) that
trains, serves, monitors, and auto-retrains regression models over the **HRI
Agricultural Harvesting dataset** (2 scenarios × 4 targets = 8 model slots,
12-algorithm tournament per slot) via MLflow with a PostgreSQL backend. The
stack is config-driven (`mlops-stack/configs/*.yaml`) — switching the YAML
swaps the whole problem (an Iris classification config is included as a
reference example). Data is versioned with DVC (local remote at
`/opt/mlops-dvc-store`), and CI/CD runs on a self-hosted runner operating
directly on the local Docker daemon (no external registry, no SSH deploy).

## Quick start

```bash
cd mlops-stack
bash setup-env.sh              # generate .env with random secrets
bash generate-certs.sh localhost
dvc pull                       # fetch data/simulation_all.csv
bash start.sh                  # build, train 8 models, start everything
```

Everything else — common commands, architecture, the 6-level QA suite,
quality gates, drift/retrain loop, observability, security — is documented in
`mlops-stack/CLAUDE.md`.

## History

The evolution is tracked in `mlops-stack/CHANGELOG.md` as "Blocks":
Block 1 (TLS + API-key auth), Block 2 (PostgreSQL + DVC), Block 3
(Prometheus + Grafana), Block 4 (generic inference API + multi-experiment).
GitHub Actions Secrets needed: `API_KEY`, `POSTGRES_PASSWORD`.
