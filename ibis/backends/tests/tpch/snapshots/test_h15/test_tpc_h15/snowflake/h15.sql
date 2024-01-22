SELECT
  "t7"."s_suppkey",
  "t7"."s_name",
  "t7"."s_address",
  "t7"."s_phone",
  "t7"."total_revenue"
FROM (
  SELECT
    "t3"."s_suppkey",
    "t3"."s_name",
    "t3"."s_address",
    "t3"."s_nationkey",
    "t3"."s_phone",
    "t3"."s_acctbal",
    "t3"."s_comment",
    "t6"."l_suppkey",
    "t6"."total_revenue"
  FROM (
    SELECT
      "t0"."S_SUPPKEY" AS "s_suppkey",
      "t0"."S_NAME" AS "s_name",
      "t0"."S_ADDRESS" AS "s_address",
      "t0"."S_NATIONKEY" AS "s_nationkey",
      "t0"."S_PHONE" AS "s_phone",
      "t0"."S_ACCTBAL" AS "s_acctbal",
      "t0"."S_COMMENT" AS "s_comment"
    FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."SUPPLIER" AS "t0"
  ) AS "t3"
  INNER JOIN (
    SELECT
      "t4"."l_suppkey",
      SUM("t4"."l_extendedprice" * (
        1 - "t4"."l_discount"
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
      FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."LINEITEM" AS "t1"
      WHERE
        "t1"."L_SHIPDATE" >= DATE_FROM_PARTS(1996, 1, 1)
        AND "t1"."L_SHIPDATE" < DATE_FROM_PARTS(1996, 4, 1)
    ) AS "t4"
    GROUP BY
      1
  ) AS "t6"
    ON "t3"."s_suppkey" = "t6"."l_suppkey"
) AS "t7"
WHERE
  "t7"."total_revenue" = (
    SELECT
      MAX("t7"."total_revenue") AS "Max(total_revenue)"
    FROM (
      SELECT
        "t3"."s_suppkey",
        "t3"."s_name",
        "t3"."s_address",
        "t3"."s_nationkey",
        "t3"."s_phone",
        "t3"."s_acctbal",
        "t3"."s_comment",
        "t6"."l_suppkey",
        "t6"."total_revenue"
      FROM (
        SELECT
          "t0"."S_SUPPKEY" AS "s_suppkey",
          "t0"."S_NAME" AS "s_name",
          "t0"."S_ADDRESS" AS "s_address",
          "t0"."S_NATIONKEY" AS "s_nationkey",
          "t0"."S_PHONE" AS "s_phone",
          "t0"."S_ACCTBAL" AS "s_acctbal",
          "t0"."S_COMMENT" AS "s_comment"
        FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."SUPPLIER" AS "t0"
      ) AS "t3"
      INNER JOIN (
        SELECT
          "t4"."l_suppkey",
          SUM("t4"."l_extendedprice" * (
            1 - "t4"."l_discount"
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
          FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."LINEITEM" AS "t1"
          WHERE
            "t1"."L_SHIPDATE" >= DATE_FROM_PARTS(1996, 1, 1)
            AND "t1"."L_SHIPDATE" < DATE_FROM_PARTS(1996, 4, 1)
        ) AS "t4"
        GROUP BY
          1
      ) AS "t6"
        ON "t3"."s_suppkey" = "t6"."l_suppkey"
    ) AS "t7"
  )
ORDER BY
  "t7"."s_suppkey" ASC