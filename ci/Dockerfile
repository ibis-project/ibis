FROM ibisproject/miniconda3

RUN apt-get -qq update -y \
 && apt-get -qq install --no-install-recommends -y git clang libboost-dev postgresql-client \
 && rm -rf /var/lib/apt/lists/* \
 && conda install conda-build --yes --channel conda-forge

ARG PYTHON

ADD ci/requirements-dev-$PYTHON.yml /tmp/requirements-dev-$PYTHON.yml
RUN conda env create -q -n "ibis_$PYTHON" -f "/tmp/requirements-dev-$PYTHON.yml"

ENV PATH /opt/conda/envs/ibis_$PYTHON/bin:$PATH

ADD . /ibis
WORKDIR /ibis

RUN pip install -e .
