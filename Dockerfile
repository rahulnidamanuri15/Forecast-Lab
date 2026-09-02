# Digest-pinned so a rebuild is reproducible. Dependabot (docker ecosystem in
# .github/dependabot.yml) bumps the digest weekly; the 3.11-slim tag stays as
# the human-readable label.
FROM python:3.11-slim@sha256:00f89b7f96f13d42900483da3253f8fb2e763eed7a0aa5f0358fec9d15d9f10c

WORKDIR /app

# No compiler needed: numpy, lightgbm and psycopg[binary] all publish manylinux
# wheels this base (bookworm, glibc 2.36) satisfies. libgomp1 is the one real
# system dep - lightgbm links OpenMP at import time and slim doesn't ship it.
# The tz database comes from the pip `tzdata` package.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Production deps only: .dockerignore excludes tests/, so pytest would have
# nothing to run here.
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Drop root before the app runs. Nothing here writes to the image at runtime -
# the API is read-only and models/ is baked in at build time - so an unwritable
# /app is correct, not a limitation. ponytail: a fixed uid, no home directory
# and no gosu; add them only if something in here ever needs to write.
RUN useradd --system --uid 10001 vericast
USER 10001

# Render (and most PaaS) inject $PORT; 8000 is the local default. Shell form in
# CMD so the variable actually expands. EXPOSE is build-time metadata and cannot
# read a runtime $PORT, so it documents the default only - the app still binds an
# override, and hosts route by their own value rather than this label.
ENV PORT=8000
EXPOSE 8000

# Reports unhealthy on the one failure a port check cannot see: the process is up
# and serving but its database is gone, which is when /health 503s. Python rather
# than curl because slim ships neither curl nor wget. sys.exit(1) on any
# exception, so a refused connection and a 503 read the same to Docker.
#
# start-period covers the pool opening on first request. Render ignores HEALTHCHECK
# (it probes over HTTP itself), so this is for `docker run` and Compose.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os,urllib.request,sys;\
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:'+os.environ['PORT']+'/health',timeout=4).status==200 else 1)" \
    || exit 1

CMD uvicorn app:app --host 0.0.0.0 --port ${PORT}
