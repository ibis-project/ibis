FROM ibisproject/miniconda3

RUN apt-get -qq update -y \
 && apt-get -qq install -y --no-install-recommends \
    git gcc make clang libboost-dev postgresql-client \
 && rm -rf /var/lib/apt/lists/*

ARG PYTHON

ADD ci/requirements-dev-$PYTHON.yml /tmp/requirements-$PYTHON.yml

RUN conda install -q -y conda-build \
 && conda install -q -y -c conda-forge python=$PYTHON \
 && conda env update -q -n root -f "/tmp/requirements-$PYTHON.yml" \
 && conda clean -a -y

ADD . /ibis
WORKDIR /ibis

RUN python setup.py develop
