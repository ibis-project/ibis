WITH t1 AS (
  SELECT
    t7."S_SUPPKEY" AS "s_suppkey",
    t7."S_NAME" AS "s_name",
    t7."S_ADDRESS" AS "s_address",
    t7."S_NATIONKEY" AS "s_nationkey",
    t7."S_PHONE" AS "s_phone",
    t7."S_ACCTBAL" AS "s_acctbal",
    t7."S_COMMENT" AS "s_comment"
  FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."SUPPLIER" AS t7
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
), t2 AS (
  SELECT
    t7."O_ORDERKEY" AS "o_orderkey",
    t7."O_CUSTKEY" AS "o_custkey",
    t7."O_ORDERSTATUS" AS "o_orderstatus",
    t7."O_TOTALPRICE" AS "o_totalprice",
    t7."O_ORDERDATE" AS "o_orderdate",
    t7."O_ORDERPRIORITY" AS "o_orderpriority",
    t7."O_CLERK" AS "o_clerk",
    t7."O_SHIPPRIORITY" AS "o_shippriority",
    t7."O_COMMENT" AS "o_comment"
  FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."ORDERS" AS t7
), t3 AS (
  SELECT
    t7."C_CUSTKEY" AS "c_custkey",
    t7."C_NAME" AS "c_name",
    t7."C_ADDRESS" AS "c_address",
    t7."C_NATIONKEY" AS "c_nationkey",
    t7."C_PHONE" AS "c_phone",
    t7."C_ACCTBAL" AS "c_acctbal",
    t7."C_MKTSEGMENT" AS "c_mktsegment",
    t7."C_COMMENT" AS "c_comment"
  FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."CUSTOMER" AS t7
), t4 AS (
  SELECT
    t7."N_NATIONKEY" AS "n_nationkey",
    t7."N_NAME" AS "n_name",
    t7."N_REGIONKEY" AS "n_regionkey",
    t7."N_COMMENT" AS "n_comment"
  FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."NATION" AS t7
), t5 AS (
  SELECT
    t4."n_name" AS "supp_nation",
    t7."n_name" AS "cust_nation",
    t0."l_shipdate" AS "l_shipdate",
    t0."l_extendedprice" AS "l_extendedprice",
    t0."l_discount" AS "l_discount",
    CAST(DATE_PART(year, t0."l_shipdate") AS SMALLINT) AS "l_year",
    t0."l_extendedprice" * (
      1 - t0."l_discount"
    ) AS "volume"
  FROM t1
  JOIN t0
    ON t1."s_suppkey" = t0."l_suppkey"
  JOIN t2
    ON t2."o_orderkey" = t0."l_orderkey"
  JOIN t3
    ON t3."c_custkey" = t2."o_custkey"
  JOIN t4
    ON t1."s_nationkey" = t4."n_nationkey"
  JOIN t4 AS t7
    ON t3."c_nationkey" = t7."n_nationkey"
)
SELECT
  t6."supp_nation",
  t6."cust_nation",
  t6."l_year",
  t6."revenue"
FROM (
  SELECT
    t5."supp_nation" AS "supp_nation",
    t5."cust_nation" AS "cust_nation",
    t5."l_year" AS "l_year",
    SUM(t5."volume") AS "revenue"
  FROM t5
  WHERE
    (
      t5."cust_nation" = 'FRANCE' AND t5."supp_nation" = 'GERMANY'
      OR t5."cust_nation" = 'GERMANY'
      AND t5."supp_nation" = 'FRANCE'
    )
    AND t5."l_shipdate" BETWEEN DATE_FROM_PARTS(1995, 1, 1) AND DATE_FROM_PARTS(1996, 12, 31)
  GROUP BY
    1,
    2,
    3
) AS t6
ORDER BY
  t6."supp_nation" ASC,
  t6."cust_nation" ASC,
  t6."l_year" ASC