#!/bin/bash
# Create the USER TEMPORARY tablespaces that ibis needs for GLOBAL TEMPORARY
# tables (used by `create_table(temp=True)` and in-memory tables).
# The Db2 image runs this from /var/custom after the database is created.
su - db2inst1 -c "db2 connect to ${DBNAME:-ibistest} && \
db2 'CREATE BUFFERPOOL IBIS_BP32K SIZE 250 PAGESIZE 32K'; \
db2 'CREATE USER TEMPORARY TABLESPACE IBIS_TEMP_4K PAGESIZE 4K MANAGED BY AUTOMATIC STORAGE'; \
db2 'CREATE USER TEMPORARY TABLESPACE IBIS_TEMP_32K PAGESIZE 32K MANAGED BY AUTOMATIC STORAGE BUFFERPOOL IBIS_BP32K'; \
db2 connect reset"
