# Digest-pinned so a rebuild is reproducible. Dependabot (docker ecosystem in
# .github/dependabot.yml) bumps the digest weekly; the 3.11-slim tag stays as
# the human-readable label.
FROM python:3.11-slim@sha256:00f89b7f96f13d42900483da3253f8fb2e763eed7a0aa5f0358fec9d15d9f10c

WORKDIR /app

# No compiler needed: numpy, lightgbm and psycopg[binary] all publish manylinux
# wheels that this base (bookworm, glibc 2.36) satisfies. libgomp1 is the one
# real system dep - lightgbm links OpenMP at import time and slim doesn't ship
# it. The tz database comes from the pip `tzdata` package, since this image has
# none and vericast/local_time.py needs Asia/Kolkata.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Production deps only. requirements-dev.txt (pytest) is deliberately not
# installed: .dockerignore excludes tests/, so the image had a test runner and
# nothing to run it on.
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Drop root before the app runs. Nothing here writes to the image at runtime -
# the API is read-only and models/ is baked in at build time - so an unwritable
# /app is correct, not a limitation. ponytail: a fixed uid, no home directory
# and no gosu; add them only if something in here ever needs to write.
RUN useradd --system --uid 10001 vericast
USER 10001

# Render (and most PaaS) inject $PORT; 8000 is the local default. Shell form so
# the variable actually expands.
#
# EXPOSE is build-time metadata and cannot read a runtime $PORT, so it documents
# the default only. If $PORT is overridden the app still binds the override -
# EXPOSE publishes nothing on its own, and every host that injects $PORT routes
# by its own value rather than by this label. Left as the honest default rather
# than an ARG that would have to be kept in sync with the deploy for no gain.
ENV PORT=8000
EXPOSE 8000
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT}
