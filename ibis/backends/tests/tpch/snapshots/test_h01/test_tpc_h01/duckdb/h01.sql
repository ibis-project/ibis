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
      CAST(1 AS TINYINT) - "t1"."l_discount"
    )) AS "sum_disc_price",
    SUM(
      (
        "t1"."l_extendedprice" * (
          CAST(1 AS TINYINT) - "t1"."l_discount"
        )
      ) * (
        "t1"."l_tax" + CAST(1 AS TINYINT)
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
      "t0"."l_quantity",
      "t0"."l_extendedprice",
      "t0"."l_discount",
      "t0"."l_tax",
      "t0"."l_returnflag",
      "t0"."l_linestatus",
      "t0"."l_shipdate",
      "t0"."l_commitdate",
      "t0"."l_receiptdate",
      "t0"."l_shipinstruct",
      "t0"."l_shipmode",
      "t0"."l_comment"
    FROM "lineitem" AS "t0"
    WHERE
      "t0"."l_shipdate" <= MAKE_DATE(1998, 9, 2)
  ) AS "t1"
  GROUP BY
    1,
    2
) AS "t2"
ORDER BY
  "t2"."l_returnflag" ASC,
  "t2"."l_linestatus" ASC