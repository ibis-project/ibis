WITH t4 AS (
  SELECT
    t7."S_SUPPKEY" AS "s_suppkey",
    t7."S_NAME" AS "s_name",
    t7."S_ADDRESS" AS "s_address",
    t7."S_NATIONKEY" AS "s_nationkey",
    t7."S_PHONE" AS "s_phone",
    t7."S_ACCTBAL" AS "s_acctbal",
    t7."S_COMMENT" AS "s_comment"
  FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."SUPPLIER" AS t7
), t3 AS (
  SELECT
    t7."N_NATIONKEY" AS "n_nationkey",
    t7."N_NAME" AS "n_name",
    t7."N_REGIONKEY" AS "n_regionkey",
    t7."N_COMMENT" AS "n_comment"
  FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."NATION" AS t7
), t1 AS (
  SELECT
    t7."PS_PARTKEY" AS "ps_partkey",
    t7."PS_SUPPKEY" AS "ps_suppkey",
    t7."PS_AVAILQTY" AS "ps_availqty",
    t7."PS_SUPPLYCOST" AS "ps_supplycost",
    t7."PS_COMMENT" AS "ps_comment"
  FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."PARTSUPP" AS t7
), t2 AS (
  SELECT
    t7."P_PARTKEY" AS "p_partkey",
    t7."P_NAME" AS "p_name",
    t7."P_MFGR" AS "p_mfgr",
    t7."P_BRAND" AS "p_brand",
    t7."P_TYPE" AS "p_type",
    t7."P_SIZE" AS "p_size",
    t7."P_CONTAINER" AS "p_container",
    t7."P_RETAILPRICE" AS "p_retailprice",
    t7."P_COMMENT" AS "p_comment"
  FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."PART" AS t7
), t0 AS (
  SELECT
    t7."L_ORDERKEY" AS "l_orderkey",
    t7."L_PARTKEY" AS "l_partkey",
    t7."L_SUPPKEY" AS "l_suppkey",
    t7."L_LINENUMBER" AS "l_linenumber",
    t7."L_QUANTITY" AS "l_quantity",
    t7."L_EXTENDEDPRICE" AS "l_extendedprice",
    t7."L_DISCOUNT" AS "l_discount",
    t7."L_TAX" AS "l_tax",
    t7."L_RETURNFLAG" AS "l_returnflag",
    t7."L_LINESTATUS" AS "l_linestatus",
    t7."L_SHIPDATE" AS "l_shipdate",
    t7."L_COMMITDATE" AS "l_commitdate",
    t7."L_RECEIPTDATE" AS "l_receiptdate",
    t7."L_SHIPINSTRUCT" AS "l_shipinstruct",
    t7."L_SHIPMODE" AS "l_shipmode",
    t7."L_COMMENT" AS "l_comment"
  FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."LINEITEM" AS t7
), t5 AS (
  SELECT
    t4."s_suppkey" AS "s_suppkey",
    t4."s_name" AS "s_name",
    t4."s_address" AS "s_address",
    t4."s_nationkey" AS "s_nationkey",
    t4."s_phone" AS "s_phone",
    t4."s_acctbal" AS "s_acctbal",
    t4."s_comment" AS "s_comment",
    t3."n_nationkey" AS "n_nationkey",
    t3."n_name" AS "n_name",
    t3."n_regionkey" AS "n_regionkey",
    t3."n_comment" AS "n_comment"
  FROM t4
  JOIN t3
    ON t4."s_nationkey" = t3."n_nationkey"
  WHERE
    t3."n_name" = 'CANADA'
    AND t4."s_suppkey" IN (
      SELECT
        t7."ps_suppkey"
      FROM (
        SELECT
          t1."ps_partkey" AS "ps_partkey",
          t1."ps_suppkey" AS "ps_suppkey",
          t1."ps_availqty" AS "ps_availqty",
          t1."ps_supplycost" AS "ps_supplycost",
          t1."ps_comment" AS "ps_comment"
        FROM t1
        WHERE
          t1."ps_partkey" IN (
            SELECT
              t8."p_partkey"
            FROM (
              SELECT
                t2."p_partkey" AS "p_partkey",
                t2."p_name" AS "p_name",
                t2."p_mfgr" AS "p_mfgr",
                t2."p_brand" AS "p_brand",
                t2."p_type" AS "p_type",
                t2."p_size" AS "p_size",
                t2."p_container" AS "p_container",
                t2."p_retailprice" AS "p_retailprice",
                t2."p_comment" AS "p_comment"
              FROM t2
              WHERE
                t2."p_name" LIKE 'forest%'
            ) AS t8
          )
          AND t1."ps_availqty" > (
            SELECT
              SUM(t0."l_quantity") AS "Sum(l_quantity)"
            FROM t0
            WHERE
              t0."l_partkey" = t1."ps_partkey"
              AND t0."l_suppkey" = t1."ps_suppkey"
              AND t0."l_shipdate" >= DATE_FROM_PARTS(1994, 1, 1)
              AND t0."l_shipdate" < DATE_FROM_PARTS(1995, 1, 1)
          ) * 0.5
      ) AS t7
    )
)
SELECT
  t6."s_name",
  t6."s_address"
FROM (
  SELECT
    t5."s_name" AS "s_name",
    t5."s_address" AS "s_address"
  FROM t5
) AS t6
ORDER BY
  t6."s_name" ASC