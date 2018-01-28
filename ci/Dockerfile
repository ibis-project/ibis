FROM ibisproject/miniconda3

RUN apt-get -qq update -y \
 && apt-get -qq install --no-install-recommends -y git clang libboost-dev postgresql-client \
 && rm -rf /var/lib/apt/lists/*

ARG python

ADD ci/requirements-dev-$python.yml /tmp/requirements-dev-$python.yml
RUN conda env create -q -n "ibis_$python" -f "/tmp/requirements-dev-$python.yml"

ENV PATH /opt/conda/envs/ibis_$python/bin:$PATH

ADD . /ibis
WORKDIR /ibis

RUN pip install -e .
