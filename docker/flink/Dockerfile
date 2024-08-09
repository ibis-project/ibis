ARG FLINK_VERSION=1.20.0
FROM flink:${FLINK_VERSION}

# ibis-flink requires PyFlink dependency
ARG FLINK_VERSION=1.20.0
RUN wget -nv -P $FLINK_HOME/lib/ https://repo1.maven.org/maven2/org/apache/flink/flink-python/${FLINK_VERSION}/flink-python-${FLINK_VERSION}.jar

# install python3 and pip3
RUN apt-get update -y && \
    apt-get install -y python3 python3-pip python3-dev openjdk-11-jdk-headless && \
    rm -rf /var/lib/apt/lists/*
RUN ln -s /usr/bin/python3 /usr/bin/python

# install PyFlink
ARG BUILDARCH
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-${BUILDARCH:-amd64}
RUN pip3 install apache-flink==${FLINK_VERSION}
