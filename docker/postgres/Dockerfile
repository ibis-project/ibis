FROM postgis/postgis:16-3.5-alpine AS pgvector-builder
RUN apk add --no-cache git build-base clang15 llvm15-dev llvm15
WORKDIR /tmp
RUN git clone --branch v0.7.4 https://github.com/pgvector/pgvector.git
WORKDIR /tmp/pgvector
RUN make && make install

FROM postgis/postgis:16-3.5-alpine
RUN apk add --no-cache postgresql16-plpython3
COPY --from=pgvector-builder /usr/local/lib/postgresql/bitcode/vector.index.bc /usr/local/lib/postgresql/bitcode/vector.index.bc
COPY --from=pgvector-builder /usr/local/lib/postgresql/vector.so /usr/local/lib/postgresql/vector.so
COPY --from=pgvector-builder /usr/local/share/postgresql/extension /usr/local/share/postgresql/extension
