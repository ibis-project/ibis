SELECT
  *
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
      "t1"."l_orderkey",
      "t1"."l_partkey",
      "t1"."l_suppkey",
      "t1"."l_linenumber",
      "t1"."l_quantity",
      "t1"."l_extendedprice",
      "t1"."l_discount",
      "t1"."l_tax",
      "t1"."l_shipdate",
      "t1"."l_commitdate",
      "t1"."l_receiptdate",
      "t1"."l_shipinstruct",
      "t1"."l_shipmode",
      "t1"."l_comment",
      "t1"."l_returnflag",
      "t1"."l_linestatus"
    FROM (
      SELECT
        *
      FROM "lineitem" AS "t0"
      WHERE
        "t0"."l_shipdate" <= DATE_TRUNC('DAY', '1998-09-02')
    ) AS "t1"
  ) AS t1
  GROUP BY
    "t1"."l_returnflag",
    "t1"."l_linestatus"
) AS "t2"
ORDER BY
  "t2"."l_returnflag" ASC,
  "t2"."l_linestatus" ASC