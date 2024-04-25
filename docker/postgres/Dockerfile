FROM postgis/postgis:15-3.3-alpine AS pgvector-builder
RUN apk add --no-cache git
RUN apk add --no-cache build-base
RUN apk add --no-cache clang15
RUN apk add --no-cache llvm15-dev llvm15
WORKDIR /tmp
RUN git clone --branch v0.6.2 https://github.com/pgvector/pgvector.git
WORKDIR /tmp/pgvector
RUN make
RUN make install

FROM postgis/postgis:15-3.3-alpine
RUN apk add --no-cache postgresql15-plpython3
COPY --from=pgvector-builder /usr/local/lib/postgresql/bitcode/vector.index.bc /usr/local/lib/postgresql/bitcode/vector.index.bc
COPY --from=pgvector-builder /usr/local/lib/postgresql/vector.so /usr/local/lib/postgresql/vector.so
COPY --from=pgvector-builder /usr/local/share/postgresql/extension /usr/local/share/postgresql/extension
