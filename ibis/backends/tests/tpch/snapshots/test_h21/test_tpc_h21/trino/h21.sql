WITH "t8" AS (
  SELECT
    "t3"."l_orderkey",
    "t3"."l_partkey",
    "t3"."l_suppkey",
    "t3"."l_linenumber",
    CAST("t3"."l_quantity" AS DECIMAL(15, 2)) AS "l_quantity",
    CAST("t3"."l_extendedprice" AS DECIMAL(15, 2)) AS "l_extendedprice",
    CAST("t3"."l_discount" AS DECIMAL(15, 2)) AS "l_discount",
    CAST("t3"."l_tax" AS DECIMAL(15, 2)) AS "l_tax",
    "t3"."l_returnflag",
    "t3"."l_linestatus",
    "t3"."l_shipdate",
    "t3"."l_commitdate",
    "t3"."l_receiptdate",
    "t3"."l_shipinstruct",
    "t3"."l_shipmode",
    "t3"."l_comment"
  FROM "hive"."ibis_sf1"."lineitem" AS "t3"
)
SELECT
  "t19"."s_name",
  "t19"."numwait"
FROM (
  SELECT
    "t18"."s_name",
    COUNT(*) AS "numwait"
  FROM (
    SELECT
      "t15"."l1_orderkey",
      "t15"."o_orderstatus",
      "t15"."l_receiptdate",
      "t15"."l_commitdate",
      "t15"."l1_suppkey",
      "t15"."s_name",
      "t15"."n_name"
    FROM (
      SELECT
        "t12"."l_orderkey" AS "l1_orderkey",
        "t10"."o_orderstatus",
        "t12"."l_receiptdate",
        "t12"."l_commitdate",
        "t12"."l_suppkey" AS "l1_suppkey",
        "t9"."s_name",
        "t7"."n_name"
      FROM (
        SELECT
          "t0"."s_suppkey",
          "t0"."s_name",
          "t0"."s_address",
          "t0"."s_nationkey",
          "t0"."s_phone",
          CAST("t0"."s_acctbal" AS DECIMAL(15, 2)) AS "s_acctbal",
          "t0"."s_comment"
        FROM "hive"."ibis_sf1"."supplier" AS "t0"
      ) AS "t9"
      INNER JOIN "t8" AS "t12"
        ON "t9"."s_suppkey" = "t12"."l_suppkey"
      INNER JOIN (
        SELECT
          "t1"."o_orderkey",
          "t1"."o_custkey",
          "t1"."o_orderstatus",
          CAST("t1"."o_totalprice" AS DECIMAL(15, 2)) AS "o_totalprice",
          "t1"."o_orderdate",
          "t1"."o_orderpriority",
          "t1"."o_clerk",
          "t1"."o_shippriority",
          "t1"."o_comment"
        FROM "hive"."ibis_sf1"."orders" AS "t1"
      ) AS "t10"
        ON "t10"."o_orderkey" = "t12"."l_orderkey"
      INNER JOIN (
        SELECT
          "t2"."n_nationkey",
          "t2"."n_name",
          "t2"."n_regionkey",
          "t2"."n_comment"
        FROM "hive"."ibis_sf1"."nation" AS "t2"
      ) AS "t7"
        ON "t9"."s_nationkey" = "t7"."n_nationkey"
    ) AS "t15"
    WHERE
      "t15"."o_orderstatus" = 'F'
      AND "t15"."l_receiptdate" > "t15"."l_commitdate"
      AND "t15"."n_name" = 'SAUDI ARABIA'
      AND EXISTS(
        SELECT
          1
        FROM "t8" AS "t13"
        WHERE
          (
            "t13"."l_orderkey" = "t15"."l1_orderkey"
          )
          AND (
            "t13"."l_suppkey" <> "t15"."l1_suppkey"
          )
      )
      AND NOT (
        EXISTS(
          SELECT
            1
          FROM "t8" AS "t14"
          WHERE
            (
              (
                "t14"."l_orderkey" = "t15"."l1_orderkey"
              )
              AND (
                "t14"."l_suppkey" <> "t15"."l1_suppkey"
              )
            )
            AND (
              "t14"."l_receiptdate" > "t14"."l_commitdate"
            )
        )
      )
  ) AS "t18"
  GROUP BY
    1
) AS "t19"
ORDER BY
  "t19"."numwait" DESC,
  "t19"."s_name" ASC
LIMIT 100