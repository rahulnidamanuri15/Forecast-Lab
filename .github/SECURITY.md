# Security Policy

## Supported versions

Only the `main` branch is supported. Fixes land there; there are no
backported release branches.

## Reporting a vulnerability

Report privately — do **not** open a public issue.

<!-- The URL says Forecast-Lab, not VeriCast, because that is still the repository
     slug; the project was renamed after the remote was created. This link has to
     match the remote or private reporting 404s, so it stays until the repo itself
     is renamed. -->

- Preferred: [GitHub private vulnerability reporting](https://github.com/rahulnidamanuri15/Forecast-Lab/security/advisories/new)
- Alternative: email rahuln152006@gmail.com with `SECURITY` in the subject.

Please include the affected file or endpoint, steps to reproduce, and the
impact you observed.

Expect an acknowledgement within 7 days and a status update at least every 14
days until the report is closed. Please give us 90 days before public
disclosure.

## Scope

In scope: this repository's code — the FastAPI service in `app.py`, the
pipeline scripts, the dashboard in `index.html`, and the GitHub Actions
workflows.

Out of scope: the third-party Open-Meteo API, the hosting provider, the
managed PostgreSQL instance, and findings that require an already-compromised
`DATABASE_URL`.

## Notes for reporters

The API is a read-only public GET service with no authentication by design.
Absence of auth is not itself a vulnerability; SQL injection, data
modification, credential leakage, or unbounded resource use are.
