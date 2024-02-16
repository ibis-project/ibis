FROM flink:1.18.1

# ibis-flink requires PyFlink dependency
RUN wget -nv -P $FLINK_HOME/lib/ https://repo1.maven.org/maven2/org/apache/flink/flink-python/1.18.1/flink-python-1.18.1.jar

# install python3, pip3, and the JDK
RUN \
  apt-get update -y && \
  apt-get install -y python3 python3-pip python3-dev git openjdk-11-jdk && \
  rm -rf /var/lib/apt/lists/* && \
  ln -s /usr/bin/python3 /usr/bin/python
ENV JAVA_HOME="/usr/lib/jvm/java-11-openjdk-arm64/"
ENV PATH="${JAVA_HOME}:${PATH}"

# install PyFlink
RUN pip3 install apache-flink==1.18.1
