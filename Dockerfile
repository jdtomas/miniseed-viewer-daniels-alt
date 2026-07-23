FROM python:3.12-slim

WORKDIR /app
ARG SQLITE_AUTOCONF_VERSION=3510300
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ca-certificates curl \
    && curl -fsSLO "https://www.sqlite.org/2026/sqlite-autoconf-${SQLITE_AUTOCONF_VERSION}.tar.gz" \
    && tar -xzf "sqlite-autoconf-${SQLITE_AUTOCONF_VERSION}.tar.gz" \
    && cd "sqlite-autoconf-${SQLITE_AUTOCONF_VERSION}" \
    && ./configure --prefix=/usr/local \
    && make -j"$(nproc)" \
    && make install \
    && ldconfig \
    && cd /app \
    && rm -rf "sqlite-autoconf-${SQLITE_AUTOCONF_VERSION}" "sqlite-autoconf-${SQLITE_AUTOCONF_VERSION}.tar.gz" \
    && apt-get purge -y --auto-remove build-essential curl \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.10.3 /uv /uvx /bin/
COPY . .
RUN uv sync --no-dev

ENV SHAKEPI_DATA_ROOT=/var/lib/shakepi/data
ENV LD_LIBRARY_PATH=/usr/local/lib
VOLUME ["/var/lib/shakepi/data"]
EXPOSE 8050
CMD ["uv", "run", "uvicorn", "shakepi.main:app_factory", "--factory", "--host", "0.0.0.0", "--port", "8050"]
