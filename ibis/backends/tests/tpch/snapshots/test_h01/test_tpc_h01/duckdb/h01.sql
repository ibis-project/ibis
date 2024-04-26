SELECT
  *
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
      *
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