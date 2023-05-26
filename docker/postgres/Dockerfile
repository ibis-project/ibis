FROM postgis/postgis:15-3.3-alpine
RUN apk add --no-cache build-base clang15 llvm15 postgresql15-plpython3 python3 py3-pip && \
    python3 -m pip install pgxnclient && \
    pgxn install vector && \
    pgxn install first_last_agg && \
    python3 -m pip uninstall -y pgxnclient && \
    rm -rf ~/.cache/pip && \
    apk del build-base clang15 llvm15 python3 py3-pip
