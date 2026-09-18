# syntax=docker/dockerfile:1
#
# The helix image, for CI. container/helix-train.def builds the SAME environment
# with apptainer; both call container/install.sh, which is the one copy of what
# gets installed.
#
# Why both exist. The clusters helix runs on have apptainer and no docker, so the
# def is the path that works anywhere with nothing but a checkout -- that
# portability was an explicit requirement and it is verified. The Dockerfile is
# what a GitHub runner can build and publish, so the cluster can pull a tested
# image instead of spending eleven minutes rebuilding one. Neither replaces the
# other; install.sh is why they cannot disagree.
#
# Pull the published result rather than building, when you can reach GHCR:
#
#     apptainer pull helix-train.sif docker://ghcr.io/omaralterkait/helix@sha256:<digest>
#
# Pin the DIGEST, not the tag. Tags are mutable rolling aliases, and an image is
# half of a lockstep pair -- "whatever :main points at today" is exactly the
# ambiguity that produced three differently-named .sif files, one of them built
# from a path source with CPU jax, while every default pointed at it.

# The pimm base, by digest. It supplies the venv, uv, torch 2.10+cu126 and the
# native wheels; helix adds pimm-data, PyWavelets and jax on top. A tag here
# would let the foundation move under a rebuild.
FROM ghcr.io/deeplearnphysics/pimm@sha256:c0dc5a64797d7176d8fc11c3cae5749008f5c6803d69e012f770d927b7a776bf

# cuda12 (default) or cpu. The image is otherwise identical either way: the jax
# wheels are prebuilt and arch-generic (PTX + SASS for sm_70..sm_90), so an image
# built on a CPU-only runner runs on Ampere unchanged. cuda12 is the default
# because helix HAS a jax GPU DSP backend and an image that cannot use it makes
# `--backend jax` a lie.
ARG JAX_EXTRA=cuda12

# install.sh reads the pimm-data revision out of pyproject.toml, so the image and
# a dev install cannot pin different revisions.
COPY container/install.sh /opt/helix-build/install.sh
COPY pyproject.toml /opt/helix-build/pyproject.toml

RUN --mount=type=cache,target=/tmp/uvcache \
    JAX_EXTRA=${JAX_EXTRA} bash /opt/helix-build/install.sh

LABEL helix.purpose="corpus building and FM training in one image" \
      helix.base="ghcr.io/deeplearnphysics/pimm v0.5.1" \
      helix.base_digest="sha256:c0dc5a64797d7176d8fc11c3cae5749008f5c6803d69e012f770d927b7a776bf" \
      helix.jax="${JAX_EXTRA}"

# helix itself is NOT baked in: it comes from a bound checkout on PYTHONPATH,
# the same way the pimm base treats pimm. pimm-data IS baked, by pinned revision,
# because the AddNoise/Digitize registration is lockstep and an environment that
# resolves two copies is the failure this image exists to prevent.
WORKDIR /work
CMD ["bash"]
