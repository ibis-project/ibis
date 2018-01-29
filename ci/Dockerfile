FROM ibisproject/miniconda3

RUN apt-get -qq update -y \
 && apt-get -qq install --no-install-recommends -y git gcc clang libboost-dev postgresql-client \
 && rm -rf /var/lib/apt/lists/*

ARG PYTHON

ADD ci/requirements-dev-$PYTHON.yml /tmp/requirements-$PYTHON.yml

RUN conda install --quiet --yes conda-build python=$PYTHON \
 && conda env update --quiet --name root --file "/tmp/requirements-$PYTHON.yml"
# && conda clean --all --yes

ADD . /ibis
WORKDIR /ibis

RUN python setup.py develop
