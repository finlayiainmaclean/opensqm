# Multi-stage image for opensqm MD / constant-pH workflows.
# Uses Pixi (same solver as CI) for reproducible conda + PyPI deps.
#
# Build (linux/amd64 required — lockfile targets linux-64):
#   docker build --platform linux/amd64 -t opensqm .
#
# Run (example):
#   docker run --rm -v "$PWD/work:/work" opensqm \
#     python -m opensqm.run_modbinddg --protein /work/protein.pdb \
#       --ligand /work/ligand.sdf --output /work/out
#
# For GPU OpenMM, swap the build base to a CUDA tag, e.g.
#   ghcr.io/prefix-dev/pixi:0.70.2-noble-cuda-12.9.1

ARG PIXI_VERSION=0.70.2
ARG CUDA_VERSION=12.9.1

FROM --platform=linux/amd64 ghcr.io/prefix-dev/pixi:${PIXI_VERSION}-noble-cuda-${CUDA_VERSION} AS build

WORKDIR /app

# Git dep (unipka) + rattler cache for faster rebuilds
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install locked deps before copying full source (layer cache)
COPY pyproject.toml pixi.lock README.md ./
RUN mkdir -p opensqm && touch opensqm/__init__.py
RUN --mount=type=cache,target=/root/.cache/rattler/cache,sharing=private \
    pixi install --locked -e default

COPY . .

RUN printf '%s\n' \
    '#!/bin/bash' \
    'export PATH="/app/.pixi/envs/default/bin:${PATH}"' \
    'export CONDA_PREFIX=/app/.pixi/envs/default' \
    'export CONDA_DEFAULT_ENV=opensqm' \
    'exec "$@"' \
    > /app/entrypoint.sh \
    && chmod +x /app/entrypoint.sh

FROM --platform=linux/amd64 ubuntu:24.04 AS production

# Runtime libs for OpenMM / AmberTools / OpenMP stacks
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        libgomp1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Keep the same prefix path as the build stage (required by shell-hook)
COPY --from=build /app/.pixi/envs/default /app/.pixi/envs/default
COPY --from=build --chmod=0755 /app/entrypoint.sh /app/entrypoint.sh
COPY --from=build /app/opensqm /app/opensqm
COPY --from=build /app/pyproject.toml /app/pyproject.toml
COPY --from=build /app/README.md /app/README.md

ENV PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["python", "-m", "opensqm.run_modbinddg", "--help"]
