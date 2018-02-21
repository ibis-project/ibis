FROM ibisproject/miniconda3

# fonts are for docs
RUN apt-get -qq update -y \
 && apt-get -qq install -y --no-install-recommends ttf-dejavu \
    git gcc make clang libboost-dev postgresql-client ca-certificates \
    openjdk-8-jre-headless ca-certificates-java \
 && rm -rf /var/lib/apt/lists/*

ARG PYTHON
ARG ENVKIND

ADD ci/requirements-${ENVKIND}-${PYTHON}.yml /

RUN conda env create -q -n ibis-${ENVKIND}-${PYTHON} -f /requirements-${ENVKIND}-${PYTHON}.yml \
 && conda install conda-build -y -q

# we intentionally keep conda artifacts in the image to speedup recipe building
# on the other hand to reduce image size run the following in the previous layer
# && conda clean -a -y

RUN echo 'source activate ibis-'${ENVKIND}-${PYTHON}' && exec "$@"' > activate.sh

ADD . /ibis
WORKDIR /ibis
RUN bash /activate.sh python setup.py develop

ENTRYPOINT ["bash", "/activate.sh"]
