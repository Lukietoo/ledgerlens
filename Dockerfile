# syntax=docker/dockerfile:1.7
# Ledgerlens API image: model baked in and checksum-verified at build time, runs as non-root.
#
#   docker build -t ledgerlens-api .
#   docker run -p 8000:8000 ledgerlens-api
ARG PYTHON=3.13-slim

# ---- deps: wheels only, cached unless pyproject changes
FROM python:${PYTHON} AS deps
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
WORKDIR /src
COPY pyproject.toml ./
# install only the dependencies first (a stub package), so code edits don't bust this layer
RUN mkdir ledgerlens && touch ledgerlens/__init__.py && pip install . && pip uninstall -y ledgerlens

# ---- app + model
FROM deps AS build
COPY ledgerlens ./ledgerlens
RUN pip install --no-deps .
# the model is a joblib pickle (loading it runs code), so fetch_model.py refuses anything whose
# SHA-256 doesn't match model.sha256 in this commit
ARG MODEL_URL=https://github.com/Lukietoo/ledgerlens/releases/download/model-v2/enricher.joblib
COPY model.sha256 ./
COPY scripts/fetch_model.py ./scripts/
RUN LEDGERLENS_MODEL_URL=${MODEL_URL} python scripts/fetch_model.py

# ---- runtime: no compilers, no pip cache, no source tree
FROM python:${PYTHON} AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin app \
    && mkdir -p /app/models   # explicit: COPY --chmod would apply 0444 to directories it creates too
COPY --from=build /opt/venv /opt/venv
COPY --from=build --chmod=0444 /src/models/enricher.joblib /app/models/enricher.joblib
COPY deploy/api-entrypoint.sh /usr/local/bin/entrypoint
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    LEDGERLENS_MODEL=/app/models/enricher.joblib \
    WEB_CONCURRENCY=1
WORKDIR /app
USER 10001
# fail the build, not the first pod, if the service user can't read the model
RUN python -c "open('/app/models/enricher.joblib','rb').read(1)"
EXPOSE 8000
ENTRYPOINT ["entrypoint"]
