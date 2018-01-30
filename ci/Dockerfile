FROM ibisproject/miniconda3

RUN apt-get -qq update -y \
 && apt-get -qq install -y --no-install-recommends \
    git gcc make clang libboost-dev postgresql-client \
 && rm -rf /var/lib/apt/lists/*

ARG PYTHON

ADD ci/requirements-dev-$PYTHON.yml /requirements-$PYTHON.yml

RUN conda install -y conda-build conda>=4.3 python=$PYTHON \
 && conda env update -n root -f /requirements-$PYTHON.yml \
 && conda clean -a -y

ADD . /ibis
WORKDIR /ibis

RUN python setup.py develop
