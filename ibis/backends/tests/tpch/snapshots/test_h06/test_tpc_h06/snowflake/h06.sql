SELECT
  SUM("t2"."l_extendedprice" * "t2"."l_discount") AS "revenue"
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
      "t1"."l_shipdate" >= DATEFROMPARTS(1994, 1, 1)
    )
    AND (
      "t1"."l_shipdate" < DATEFROMPARTS(1995, 1, 1)
    )
    AND "t1"."l_discount" BETWEEN 0.05 AND 0.07
    AND (
      "t1"."l_quantity" < 24
    )
) AS "t2"