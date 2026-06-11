FROM staphb/snippy:4.6.0

USER root
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    bzip2 \
    git \
    ncbi-blast+ \
    diamond-aligner \
    prodigal \
    && rm -rf /var/lib/apt/lists/*

RUN if ! command -v diamond >/dev/null 2>&1 && command -v diamond-aligner >/dev/null 2>&1; then \
      ln -s "$(command -v diamond-aligner)" /usr/local/bin/diamond; \
    fi

ENV MAMBA_ROOT_PREFIX=/opt/micromamba
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
    | tar -xvj -C /usr/local/bin --strip-components=1 bin/micromamba \
    && micromamba create -y -n app -c conda-forge python=3.11 pip \
    && micromamba clean --all --yes

COPY requirements.txt /tmp/requirements.txt
RUN micromamba run -n app python -m pip install --no-cache-dir --upgrade pip \
    && micromamba run -n app python -m pip install --no-cache-dir -r /tmp/requirements.txt

RUN micromamba create -y -n rgi -c conda-forge -c bioconda python=3.10 rgi \
    && micromamba clean --all --yes

RUN printf '#!/bin/sh\nexec micromamba run -n rgi rgi "$@"\n' > /usr/local/bin/rgi \
    && chmod +x /usr/local/bin/rgi

COPY . /app

RUN rgi load --card_json /app/assets/card/card.json --local

ENV PATH="/opt/micromamba/envs/app/bin:${PATH}"
ENV APP_THREADS=2
ENV LOKY_MAX_CPU_COUNT=1
ENV OMP_NUM_THREADS=1
ENV PYTHONUNBUFFERED=1

EXPOSE 7860

CMD ["micromamba", "run", "-n", "app", "python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
