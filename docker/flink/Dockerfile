FROM flink:1.18.1

# ibis-flink requires PyFlink dependency
RUN wget -nv -P $FLINK_HOME/lib/ https://repo1.maven.org/maven2/org/apache/flink/flink-python/1.18.1/flink-python-1.18.1.jar

# install python3 and pip3
RUN apt-get update -y && \
apt-get install -y python3 python3-pip python3-dev && rm -rf /var/lib/apt/lists/*
RUN ln -s /usr/bin/python3 /usr/bin/python

# install PyFlink
RUN pip3 install apache-flink==1.18.1
