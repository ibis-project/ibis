ARG FLINK_VERSION=2.3.0
FROM flink:${FLINK_VERSION}

# install python3 and venv support; the base image is Ubuntu, whose system
# python is marked externally managed (PEP 668), so PyFlink goes in a venv
USER root
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends python3 python3-venv && \
    rm -rf /var/lib/apt/lists/*
USER flink

# ibis-flink requires the PyFlink jar, which the distribution ships in opt/
ARG FLINK_VERSION=2.3.0
RUN cp $FLINK_HOME/opt/flink-python-${FLINK_VERSION}.jar $FLINK_HOME/lib/

# install PyFlink
ENV PATH=$FLINK_HOME/venv/bin:$PATH
RUN python3 -m venv $FLINK_HOME/venv && \
    pip install --no-cache-dir apache-flink==${FLINK_VERSION}
