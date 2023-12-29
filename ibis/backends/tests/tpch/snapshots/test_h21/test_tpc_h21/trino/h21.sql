SELECT
  "t21"."s_name",
  "t21"."numwait"
FROM (
  SELECT
    "t20"."s_name",
    COUNT(*) AS "numwait"
  FROM (
    SELECT
      "t17"."l1_orderkey",
      "t17"."o_orderstatus",
      "t17"."l_receiptdate",
      "t17"."l_commitdate",
      "t17"."l1_suppkey",
      "t17"."s_name",
      "t17"."n_name"
    FROM (
      SELECT
        "t10"."l_orderkey" AS "l1_orderkey",
        "t13"."o_orderstatus",
        "t10"."l_receiptdate",
        "t10"."l_commitdate",
        "t10"."l_suppkey" AS "l1_suppkey",
        "t9"."s_name",
        "t8"."n_name"
      FROM (
        SELECT
          "t0"."s_suppkey",
          "t0"."s_name",
          "t0"."s_address",
          "t0"."s_nationkey",
          "t0"."s_phone",
          CAST("t0"."s_acctbal" AS DECIMAL(15, 2)) AS "s_acctbal",
          "t0"."s_comment"
        FROM "supplier" AS "t0"
      ) AS "t9"
      INNER JOIN (
        SELECT
          "t1"."l_orderkey",
          "t1"."l_partkey",
          "t1"."l_suppkey",
          "t1"."l_linenumber",
          CAST("t1"."l_quantity" AS DECIMAL(15, 2)) AS "l_quantity",
          CAST("t1"."l_extendedprice" AS DECIMAL(15, 2)) AS "l_extendedprice",
          CAST("t1"."l_discount" AS DECIMAL(15, 2)) AS "l_discount",
          CAST("t1"."l_tax" AS DECIMAL(15, 2)) AS "l_tax",
          "t1"."l_returnflag",
          "t1"."l_linestatus",
          "t1"."l_shipdate",
          "t1"."l_commitdate",
          "t1"."l_receiptdate",
          "t1"."l_shipinstruct",
          "t1"."l_shipmode",
          "t1"."l_comment"
        FROM "lineitem" AS "t1"
      ) AS "t10"
        ON "t9"."s_suppkey" = "t10"."l_suppkey"
      INNER JOIN (
        SELECT
          "t2"."o_orderkey",
          "t2"."o_custkey",
          "t2"."o_orderstatus",
          CAST("t2"."o_totalprice" AS DECIMAL(15, 2)) AS "o_totalprice",
          "t2"."o_orderdate",
          "t2"."o_orderpriority",
          "t2"."o_clerk",
          "t2"."o_shippriority",
          "t2"."o_comment"
        FROM "orders" AS "t2"
      ) AS "t13"
        ON "t13"."o_orderkey" = "t10"."l_orderkey"
      INNER JOIN (
        SELECT
          "t3"."n_nationkey",
          "t3"."n_name",
          "t3"."n_regionkey",
          "t3"."n_comment"
        FROM "nation" AS "t3"
      ) AS "t8"
        ON "t9"."s_nationkey" = "t8"."n_nationkey"
    ) AS "t17"
    WHERE
      "t17"."o_orderstatus" = 'F'
      AND "t17"."l_receiptdate" > "t17"."l_commitdate"
      AND "t17"."n_name" = 'SAUDI ARABIA'
      AND EXISTS(
        SELECT
          1 AS "1"
        FROM (
          SELECT
            "t1"."l_orderkey",
            "t1"."l_partkey",
            "t1"."l_suppkey",
            "t1"."l_linenumber",
            CAST("t1"."l_quantity" AS DECIMAL(15, 2)) AS "l_quantity",
            CAST("t1"."l_extendedprice" AS DECIMAL(15, 2)) AS "l_extendedprice",
            CAST("t1"."l_discount" AS DECIMAL(15, 2)) AS "l_discount",
            CAST("t1"."l_tax" AS DECIMAL(15, 2)) AS "l_tax",
            "t1"."l_returnflag",
            "t1"."l_linestatus",
            "t1"."l_shipdate",
            "t1"."l_commitdate",
            "t1"."l_receiptdate",
            "t1"."l_shipinstruct",
            "t1"."l_shipmode",
            "t1"."l_comment"
          FROM "lineitem" AS "t1"
        ) AS "t11"
        WHERE
          (
            "t11"."l_orderkey" = "t17"."l1_orderkey"
          )
          AND (
            "t11"."l_suppkey" <> "t17"."l1_suppkey"
          )
      )
      AND NOT (
        EXISTS(
          SELECT
            1 AS "1"
          FROM (
            SELECT
              "t1"."l_orderkey",
              "t1"."l_partkey",
              "t1"."l_suppkey",
              "t1"."l_linenumber",
              CAST("t1"."l_quantity" AS DECIMAL(15, 2)) AS "l_quantity",
              CAST("t1"."l_extendedprice" AS DECIMAL(15, 2)) AS "l_extendedprice",
              CAST("t1"."l_discount" AS DECIMAL(15, 2)) AS "l_discount",
              CAST("t1"."l_tax" AS DECIMAL(15, 2)) AS "l_tax",
              "t1"."l_returnflag",
              "t1"."l_linestatus",
              "t1"."l_shipdate",
              "t1"."l_commitdate",
              "t1"."l_receiptdate",
              "t1"."l_shipinstruct",
              "t1"."l_shipmode",
              "t1"."l_comment"
            FROM "lineitem" AS "t1"
          ) AS "t12"
          WHERE
            (
              (
                "t12"."l_orderkey" = "t17"."l1_orderkey"
              )
              AND (
                "t12"."l_suppkey" <> "t17"."l1_suppkey"
              )
            )
            AND (
              "t12"."l_receiptdate" > "t12"."l_commitdate"
            )
        )
      )
  ) AS "t20"
  GROUP BY
    1
) AS "t21"
ORDER BY
  "t21"."numwait" DESC,
  "t21"."s_name" ASC
LIMIT 100