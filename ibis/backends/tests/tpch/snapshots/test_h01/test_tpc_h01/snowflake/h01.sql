SELECT
  "t2"."l_returnflag",
  "t2"."l_linestatus",
  "t2"."sum_qty",
  "t2"."sum_base_price",
  "t2"."sum_disc_price",
  "t2"."sum_charge",
  "t2"."avg_qty",
  "t2"."avg_price",
  "t2"."avg_disc",
  "t2"."count_order"
FROM (
  SELECT
    "t1"."l_returnflag",
    "t1"."l_linestatus",
    SUM("t1"."l_quantity") AS "sum_qty",
    SUM("t1"."l_extendedprice") AS "sum_base_price",
    SUM("t1"."l_extendedprice" * (
      1 - "t1"."l_discount"
    )) AS "sum_disc_price",
    SUM(
      (
        "t1"."l_extendedprice" * (
          1 - "t1"."l_discount"
        )
      ) * (
        "t1"."l_tax" + 1
      )
    ) AS "sum_charge",
    AVG("t1"."l_quantity") AS "avg_qty",
    AVG("t1"."l_extendedprice") AS "avg_price",
    AVG("t1"."l_discount") AS "avg_disc",
    COUNT(*) AS "count_order"
  FROM (
    SELECT
      "t0"."L_ORDERKEY" AS "l_orderkey",
      "t0"."L_PARTKEY" AS "l_partkey",
      "t0"."L_SUPPKEY" AS "l_suppkey",
      "t0"."L_LINENUMBER" AS "l_linenumber",
      "t0"."L_QUANTITY" AS "l_quantity",
      "t0"."L_EXTENDEDPRICE" AS "l_extendedprice",
      "t0"."L_DISCOUNT" AS "l_discount",
      "t0"."L_TAX" AS "l_tax",
      "t0"."L_RETURNFLAG" AS "l_returnflag",
      "t0"."L_LINESTATUS" AS "l_linestatus",
      "t0"."L_SHIPDATE" AS "l_shipdate",
      "t0"."L_COMMITDATE" AS "l_commitdate",
      "t0"."L_RECEIPTDATE" AS "l_receiptdate",
      "t0"."L_SHIPINSTRUCT" AS "l_shipinstruct",
      "t0"."L_SHIPMODE" AS "l_shipmode",
      "t0"."L_COMMENT" AS "l_comment"
    FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."LINEITEM" AS "t0"
    WHERE
      "t0"."L_SHIPDATE" <= DATE_FROM_PARTS(1998, 9, 2)
  ) AS "t1"
  GROUP BY
    1,
    2
) AS "t2"
ORDER BY
  "t2"."l_returnflag" ASC,
  "t2"."l_linestatus" ASC