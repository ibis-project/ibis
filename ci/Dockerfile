FROM ibisproject/miniconda3

RUN apt-get -qq update -y \
 && apt-get -qq install -y --no-install-recommends \
    git gcc make clang libboost-dev postgresql-client \
 && rm -rf /var/lib/apt/lists/*

ADD ci/requirements-dev-2.7.yml \
    ci/requirements-dev-3.4.yml \
    ci/requirements-dev-3.5.yml \
    ci/requirements-dev-3.6.yml /

RUN conda env create -q -n ibis-2.7 -f /requirements-dev-2.7.yml \
 && conda env create -q -n ibis-3.4 -f /requirements-dev-3.4.yml \
 && conda env create -q -n ibis-3.5 -f /requirements-dev-3.5.yml \
 && conda env create -q -n ibis-3.6 -f /requirements-dev-3.6.yml \
 && conda install conda-build -y -q \
 && conda clean -a -y

ADD . /ibis
WORKDIR /ibis

RUN /bin/bash -c "source activate ibis-$PYTHON"

RUN pip install -e .

