FROM ibisproject/miniconda3

RUN apt-get -qq update -y \
 && apt-get -qq install -y --no-install-recommends \
    git gcc make clang libboost-dev postgresql-client \
 && rm -rf /var/lib/apt/lists/*

ARG PYTHON

ADD ci/requirements-dev-$PYTHON.yml /

RUN conda env create -q -n ibis-$PYTHON -f /requirements-dev-$PYTHON.yml \
 && conda install conda-build -y -q \
 && conda clean -a -y

RUN echo 'source activate ibis-'$PYTHON' && exec "$@"' > activate.sh

ADD . /ibis
WORKDIR /ibis
RUN bash /activate.sh python setup.py develop

ENTRYPOINT ["bash", "/activate.sh"]
