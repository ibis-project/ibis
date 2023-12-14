SELECT
  "t7"."s_suppkey" AS "s_suppkey",
  "t7"."s_name" AS "s_name",
  "t7"."s_address" AS "s_address",
  "t7"."s_phone" AS "s_phone",
  "t7"."total_revenue" AS "total_revenue"
FROM (
  SELECT
    "t2"."s_suppkey" AS "s_suppkey",
    "t2"."s_name" AS "s_name",
    "t2"."s_address" AS "s_address",
    "t2"."s_nationkey" AS "s_nationkey",
    "t2"."s_phone" AS "s_phone",
    "t2"."s_acctbal" AS "s_acctbal",
    "t2"."s_comment" AS "s_comment",
    "t5"."l_suppkey" AS "l_suppkey",
    "t5"."total_revenue" AS "total_revenue"
  FROM (
    SELECT
      "t0"."S_SUPPKEY" AS "s_suppkey",
      "t0"."S_NAME" AS "s_name",
      "t0"."S_ADDRESS" AS "s_address",
      "t0"."S_NATIONKEY" AS "s_nationkey",
      "t0"."S_PHONE" AS "s_phone",
      "t0"."S_ACCTBAL" AS "s_acctbal",
      "t0"."S_COMMENT" AS "s_comment"
    FROM "SUPPLIER" AS "t0"
  ) AS "t2"
  INNER JOIN (
    SELECT
      "t3"."l_suppkey" AS "l_suppkey",
      SUM("t3"."l_extendedprice" * (
        1 - "t3"."l_discount"
      )) AS "total_revenue"
    FROM (
      SELECT
        "t1"."L_ORDERKEY" AS "l_orderkey",
        "t1"."L_PARTKEY" AS "l_partkey",
        "t1"."L_SUPPKEY" AS "l_suppkey",
        "t1"."L_LINENUMBER" AS "l_linenumber",
        "t1"."L_QUANTITY" AS "l_quantity",
        "t1"."L_EXTENDEDPRICE" AS "l_extendedprice",
        "t1"."L_DISCOUNT" AS "l_discount",
        "t1"."L_TAX" AS "l_tax",
        "t1"."L_RETURNFLAG" AS "l_returnflag",
        "t1"."L_LINESTATUS" AS "l_linestatus",
        "t1"."L_SHIPDATE" AS "l_shipdate",
        "t1"."L_COMMITDATE" AS "l_commitdate",
        "t1"."L_RECEIPTDATE" AS "l_receiptdate",
        "t1"."L_SHIPINSTRUCT" AS "l_shipinstruct",
        "t1"."L_SHIPMODE" AS "l_shipmode",
        "t1"."L_COMMENT" AS "l_comment"
      FROM "LINEITEM" AS "t1"
      WHERE
        "t1"."L_SHIPDATE" >= DATEFROMPARTS(1996, 1, 1)
        AND "t1"."L_SHIPDATE" < DATEFROMPARTS(1996, 4, 1)
    ) AS "t3"
    GROUP BY
      1
  ) AS "t5"
    ON "t2"."s_suppkey" = "t5"."l_suppkey"
) AS "t7"
WHERE
  "t7"."total_revenue" = (
    SELECT
      MAX("t7"."total_revenue") AS "Max(total_revenue)"
    FROM (
      SELECT
        "t2"."s_suppkey" AS "s_suppkey",
        "t2"."s_name" AS "s_name",
        "t2"."s_address" AS "s_address",
        "t2"."s_nationkey" AS "s_nationkey",
        "t2"."s_phone" AS "s_phone",
        "t2"."s_acctbal" AS "s_acctbal",
        "t2"."s_comment" AS "s_comment",
        "t5"."l_suppkey" AS "l_suppkey",
        "t5"."total_revenue" AS "total_revenue"
      FROM (
        SELECT
          "t0"."S_SUPPKEY" AS "s_suppkey",
          "t0"."S_NAME" AS "s_name",
          "t0"."S_ADDRESS" AS "s_address",
          "t0"."S_NATIONKEY" AS "s_nationkey",
          "t0"."S_PHONE" AS "s_phone",
          "t0"."S_ACCTBAL" AS "s_acctbal",
          "t0"."S_COMMENT" AS "s_comment"
        FROM "SUPPLIER" AS "t0"
      ) AS "t2"
      INNER JOIN (
        SELECT
          "t3"."l_suppkey" AS "l_suppkey",
          SUM("t3"."l_extendedprice" * (
            1 - "t3"."l_discount"
          )) AS "total_revenue"
        FROM (
          SELECT
            "t1"."L_ORDERKEY" AS "l_orderkey",
            "t1"."L_PARTKEY" AS "l_partkey",
            "t1"."L_SUPPKEY" AS "l_suppkey",
            "t1"."L_LINENUMBER" AS "l_linenumber",
            "t1"."L_QUANTITY" AS "l_quantity",
            "t1"."L_EXTENDEDPRICE" AS "l_extendedprice",
            "t1"."L_DISCOUNT" AS "l_discount",
            "t1"."L_TAX" AS "l_tax",
            "t1"."L_RETURNFLAG" AS "l_returnflag",
            "t1"."L_LINESTATUS" AS "l_linestatus",
            "t1"."L_SHIPDATE" AS "l_shipdate",
            "t1"."L_COMMITDATE" AS "l_commitdate",
            "t1"."L_RECEIPTDATE" AS "l_receiptdate",
            "t1"."L_SHIPINSTRUCT" AS "l_shipinstruct",
            "t1"."L_SHIPMODE" AS "l_shipmode",
            "t1"."L_COMMENT" AS "l_comment"
          FROM "LINEITEM" AS "t1"
          WHERE
            "t1"."L_SHIPDATE" >= DATEFROMPARTS(1996, 1, 1)
            AND "t1"."L_SHIPDATE" < DATEFROMPARTS(1996, 4, 1)
        ) AS "t3"
        GROUP BY
          1
      ) AS "t5"
        ON "t2"."s_suppkey" = "t5"."l_suppkey"
    ) AS "t7"
  )
ORDER BY
  "t7"."s_suppkey" ASC