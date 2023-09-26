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
)
SELECT
  t1."l_returnflag",
  t1."l_linestatus",
  t1."sum_qty",
  t1."sum_base_price",
  t1."sum_disc_price",
  t1."sum_charge",
  t1."avg_qty",
  t1."avg_price",
  t1."avg_disc",
  t1."count_order"
FROM (
  SELECT
    t0."l_returnflag" AS "l_returnflag",
    t0."l_linestatus" AS "l_linestatus",
    SUM(t0."l_quantity") AS "sum_qty",
    SUM(t0."l_extendedprice") AS "sum_base_price",
    SUM(t0."l_extendedprice" * (
      1 - t0."l_discount"
    )) AS "sum_disc_price",
    SUM(t0."l_extendedprice" * (
      1 - t0."l_discount"
    ) * (
      t0."l_tax" + 1
    )) AS "sum_charge",
    AVG(t0."l_quantity") AS "avg_qty",
    AVG(t0."l_extendedprice") AS "avg_price",
    AVG(t0."l_discount") AS "avg_disc",
    COUNT(*) AS "count_order"
  FROM t0
  WHERE
    t0."l_shipdate" <= DATE_FROM_PARTS(1998, 9, 2)
  GROUP BY
    1,
    2
) AS t1
ORDER BY
  t1."l_returnflag" ASC,
  t1."l_linestatus" ASC