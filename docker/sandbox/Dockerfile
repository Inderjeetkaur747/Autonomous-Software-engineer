# Sandbox image: runs a repository's tests. It is started with no network.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt \
    && useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin sandbox

USER 10001:10001
WORKDIR /work

# The repo is mounted read-only at /src. Copy it (without .git) into /work, then run the
# command we were given. "$@" is exactly the arguments after the image name, no shell parsing.
ENTRYPOINT ["sh", "-c", "set -eu; mkdir -p /work; cd /src; find . -mindepth 1 -name .git -prune -o -print0 | tar --null --no-recursion -T - -cf - | tar -C /work -xmf -; cd /work; exec \"$@\"", "sandbox"]
