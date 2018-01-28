FROM ibisproject/miniconda3

ARG python

ADD ci/requirements-dev-$python.yml /tmp/requirements-dev-$python.yml
RUN conda env create -q -n "ibis_$python" -f "/tmp/requirements-dev-$python.yml"

ENV PATH /opt/conda/envs/ibis_$python/bin:$PATH

ADD . /ibis
WORKDIR /ibis

RUN pip install -e .
