# Changelog

All notable changes to VeriCast. Format follows Keep a Changelog; versioning is SemVer.

## [1.0.0] - 2026-09-15

### Fixed

- `source=latest` documented as newest issuance by `(forecast_date, created_at)`, never backtest; removed stale "daily preferred on tied dates" wording.
- `DATABASE_URL` read lazily via `get_database_url()` so `import app` no longer crashes tooling without a DSN; direct connections enforce the same 10s `statement_timeout` as the pool.
- `/predictions` and `/electricity/predictions` return `total` alongside `count` and expose a `PredictionPage` response model for reliable pagination.
- CI, daily pipeline, weekly retrain, and readiness gate install `requirements.lock` for reproducible trees; Docker image stays on `requirements.txt` (same direct pins, no dev deps) with sync guidance.
- Coverage gate raised 60 to 70; documented path to 80.
- Retrain promotion bar tightened 1% to 5% over persistence (`MAX_BASELINE_RATIO 0.99 -> 0.95`).
- Docker defaults to `--workers 1` to keep the per-process 120 req/min limiter exact.
- Documented intentional `PM25_MIN=1.0` (0.0 is an upstream empty-field sentinel, not clean air).

### Added

- `docker-compose.yml` local stack (API + Postgres 17, matching CI).
- `CHANGELOG.md` and rewritten professional `README.md`: TOC, live demo links, env-var table, rate-limit/cache docs, troubleshooting, roadmap; corrected 205 to 219 tests and bundle-authoritative artifact docs.

## [0.1.0] - earlier

- Published-then-verified pipelines for Nagpur PM2.5 and Maharashtra demand with provenance split, gated retraining, and 23-check readiness gate.
