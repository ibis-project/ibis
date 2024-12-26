FROM postgis/postgis:17-3.5-alpine AS pgvector-builder
RUN apk add --no-cache git build-base clang19 llvm19-dev llvm19
WORKDIR /tmp
RUN git clone --branch v0.8.0 https://github.com/pgvector/pgvector.git
WORKDIR /tmp/pgvector
RUN make && make install

FROM postgis/postgis:17-3.5-alpine
RUN apk add --no-cache postgresql17-plpython3
COPY --from=pgvector-builder /usr/local/lib/postgresql/bitcode/vector.index.bc /usr/local/lib/postgresql/bitcode/vector.index.bc
COPY --from=pgvector-builder /usr/local/lib/postgresql/vector.so /usr/local/lib/postgresql/vector.so
COPY --from=pgvector-builder /usr/local/share/postgresql/extension /usr/local/share/postgresql/extension
