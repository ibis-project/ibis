FROM ibisproject/miniconda3

RUN apt-get -qq update -y \
 && apt-get -qq install --no-install-recommends -y git clang libboost-dev postgresql-client \
 && rm -rf /var/lib/apt/lists/*

ARG PYTHON

ADD ci/requirements-dev-$PYTHON.yml /tmp/requirements.yml
RUN conda env update --name root --quiet --file "/tmp/requirements.yml" \
 && conda install --yes --quiet --channel conda-forge conda-build \
 && conda clean --all --yes

ADD . /ibis
WORKDIR /ibis

RUN pip install -e .
