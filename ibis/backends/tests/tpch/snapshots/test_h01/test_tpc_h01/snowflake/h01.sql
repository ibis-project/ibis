SELECT
  *
FROM (
  SELECT
    "t2"."l_returnflag" AS "l_returnflag",
    "t2"."l_linestatus" AS "l_linestatus",
    SUM("t2"."l_quantity") AS "sum_qty",
    SUM("t2"."l_extendedprice") AS "sum_base_price",
    SUM("t2"."l_extendedprice" * (
      1 - "t2"."l_discount"
    )) AS "sum_disc_price",
    SUM(
      (
        "t2"."l_extendedprice" * (
          1 - "t2"."l_discount"
        )
      ) * (
        "t2"."l_tax" + 1
      )
    ) AS "sum_charge",
    AVG("t2"."l_quantity") AS "avg_qty",
    AVG("t2"."l_extendedprice") AS "avg_price",
    AVG("t2"."l_discount") AS "avg_disc",
    COUNT(*) AS "count_order"
  FROM (
    SELECT
      *
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
      FROM "LINEITEM" AS "t0"
    ) AS "t1"
    WHERE
      (
        "t1"."l_shipdate" <= DATEFROMPARTS(1998, 9, 2)
      )
  ) AS "t2"
  GROUP BY
    1,
    2
) AS "t3"
ORDER BY
  "t3"."l_returnflag" ASC,
  "t3"."l_linestatus" ASC