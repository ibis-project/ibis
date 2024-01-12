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
      "t0"."l_orderkey",
      "t0"."l_partkey",
      "t0"."l_suppkey",
      "t0"."l_linenumber",
      CAST("t0"."l_quantity" AS DECIMAL(15, 2)) AS "l_quantity",
      CAST("t0"."l_extendedprice" AS DECIMAL(15, 2)) AS "l_extendedprice",
      CAST("t0"."l_discount" AS DECIMAL(15, 2)) AS "l_discount",
      CAST("t0"."l_tax" AS DECIMAL(15, 2)) AS "l_tax",
      "t0"."l_returnflag",
      "t0"."l_linestatus",
      "t0"."l_shipdate",
      "t0"."l_commitdate",
      "t0"."l_receiptdate",
      "t0"."l_shipinstruct",
      "t0"."l_shipmode",
      "t0"."l_comment"
    FROM "hive"."ibis_sf1"."lineitem" AS "t0"
    WHERE
      "t0"."l_shipdate" <= FROM_ISO8601_DATE('1998-09-02')
  ) AS "t1"
  GROUP BY
    1,
    2
) AS "t2"
ORDER BY
  "t2"."l_returnflag" ASC,
  "t2"."l_linestatus" ASC