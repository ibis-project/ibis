WITH t0 AS (
  SELECT
    t2."L_ORDERKEY" AS "l_orderkey",
    t2."L_PARTKEY" AS "l_partkey",
    t2."L_SUPPKEY" AS "l_suppkey",
    t2."L_LINENUMBER" AS "l_linenumber",
    t2."L_QUANTITY" AS "l_quantity",
    t2."L_EXTENDEDPRICE" AS "l_extendedprice",
    t2."L_DISCOUNT" AS "l_discount",
    t2."L_TAX" AS "l_tax",
    t2."L_RETURNFLAG" AS "l_returnflag",
    t2."L_LINESTATUS" AS "l_linestatus",
    t2."L_SHIPDATE" AS "l_shipdate",
    t2."L_COMMITDATE" AS "l_commitdate",
    t2."L_RECEIPTDATE" AS "l_receiptdate",
    t2."L_SHIPINSTRUCT" AS "l_shipinstruct",
    t2."L_SHIPMODE" AS "l_shipmode",
    t2."L_COMMENT" AS "l_comment"
  FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."LINEITEM" AS t2
), t1 AS (
  SELECT
    t2."P_PARTKEY" AS "p_partkey",
    t2."P_NAME" AS "p_name",
    t2."P_MFGR" AS "p_mfgr",
    t2."P_BRAND" AS "p_brand",
    t2."P_TYPE" AS "p_type",
    t2."P_SIZE" AS "p_size",
    t2."P_CONTAINER" AS "p_container",
    t2."P_RETAILPRICE" AS "p_retailprice",
    t2."P_COMMENT" AS "p_comment"
  FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."PART" AS t2
)
SELECT
  SUM(t0."l_extendedprice" * (
    1 - t0."l_discount"
  )) AS "revenue"
FROM t0
JOIN t1
  ON t1."p_partkey" = t0."l_partkey"
WHERE
  t1."p_brand" = 'Brand#12'
  AND t1."p_container" IN ('SM CASE', 'SM BOX', 'SM PACK', 'SM PKG')
  AND t0."l_quantity" >= 1
  AND t0."l_quantity" <= 11
  AND t1."p_size" BETWEEN 1 AND 5
  AND t0."l_shipmode" IN ('AIR', 'AIR REG')
  AND t0."l_shipinstruct" = 'DELIVER IN PERSON'
  OR t1."p_brand" = 'Brand#23'
  AND t1."p_container" IN ('MED BAG', 'MED BOX', 'MED PKG', 'MED PACK')
  AND t0."l_quantity" >= 10
  AND t0."l_quantity" <= 20
  AND t1."p_size" BETWEEN 1 AND 10
  AND t0."l_shipmode" IN ('AIR', 'AIR REG')
  AND t0."l_shipinstruct" = 'DELIVER IN PERSON'
  OR t1."p_brand" = 'Brand#34'
  AND t1."p_container" IN ('LG CASE', 'LG BOX', 'LG PACK', 'LG PKG')
  AND t0."l_quantity" >= 20
  AND t0."l_quantity" <= 30
  AND t1."p_size" BETWEEN 1 AND 15
  AND t0."l_shipmode" IN ('AIR', 'AIR REG')
  AND t0."l_shipinstruct" = 'DELIVER IN PERSON'