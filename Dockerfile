# Digest-pinned so a rebuild is reproducible. Dependabot (docker ecosystem in
# .github/dependabot.yml) bumps the digest weekly; the 3.11-slim tag stays as
# the human-readable label.
FROM python:3.11-slim@sha256:00f89b7f96f13d42900483da3253f8fb2e763eed7a0aa5f0358fec9d15d9f10c

WORKDIR /app

# No compiler needed: numpy, lightgbm and psycopg[binary] all publish manylinux
# wheels that this base (bookworm, glibc 2.36) satisfies. libgomp1 is the one
# real system dep - lightgbm links OpenMP at import time and slim doesn't ship
# it. The tz database comes from the pip `tzdata` package, since this image has
# none and local_time.py needs Asia/Kolkata.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render (and most PaaS) inject $PORT; 8000 is the local default. Shell form so
# the variable actually expands.
ENV PORT=8000
EXPOSE 8000
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT}
