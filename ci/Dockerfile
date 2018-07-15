FROM ibisproject/miniconda3

ADD ci/cloudera.list /etc/apt/sources.list.d/

# fonts are for docs
RUN apt-get -qq update --yes \
 && apt-get -qq install --yes --allow-unauthenticated --no-install-recommends \
   build-essential ttf-dejavu \
   git make clang libboost-dev postgresql-client ca-certificates \
 && rm -rf /var/lib/apt/lists/*

ARG PYTHON
ARG ENVKIND

COPY ci/requirements-${ENVKIND}-${PYTHON}.yml /

RUN conda env create -q -n ibis-${ENVKIND}-${PYTHON} -f /requirements-${ENVKIND}-${PYTHON}.yml \
 && conda install conda-build -y -q

# we intentionally keep conda artifacts in the image to speedup recipe building
# on the other hand to reduce image size run the following in the previous layer
# && conda clean -a -y

RUN echo 'source activate ibis-'${ENVKIND}-${PYTHON}' && exec "$@"' > activate.sh

COPY . /ibis
WORKDIR /ibis

RUN bash /activate.sh python setup.py develop

ENTRYPOINT ["bash", "/activate.sh"]
