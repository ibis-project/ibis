SELECT
  "t9"."s_name",
  "t9"."s_address"
FROM (
  SELECT
    "t5"."s_suppkey",
    "t5"."s_name",
    "t5"."s_address",
    "t5"."s_nationkey",
    "t5"."s_phone",
    "t5"."s_acctbal",
    "t5"."s_comment",
    "t6"."n_nationkey",
    "t6"."n_name",
    "t6"."n_regionkey",
    "t6"."n_comment"
  FROM "supplier" AS "t5"
  INNER JOIN "nation" AS "t6"
    ON "t5"."s_nationkey" = "t6"."n_nationkey"
) AS "t9"
WHERE
  "t9"."n_name" = 'CANADA'
  AND "t9"."s_suppkey" IN (
    SELECT
      "t1"."ps_suppkey"
    FROM "partsupp" AS "t1"
    WHERE
      "t1"."ps_partkey" IN (
        SELECT
          "t3"."p_partkey"
        FROM "part" AS "t3"
        WHERE
          "t3"."p_name" LIKE 'forest%'
      )
      AND "t1"."ps_availqty" > (
        (
          SELECT
            SUM("t8"."l_quantity") AS "Sum(l_quantity)"
          FROM (
            SELECT
              "t4"."l_orderkey",
              "t4"."l_partkey",
              "t4"."l_suppkey",
              "t4"."l_linenumber",
              "t4"."l_quantity",
              "t4"."l_extendedprice",
              "t4"."l_discount",
              "t4"."l_tax",
              "t4"."l_returnflag",
              "t4"."l_linestatus",
              "t4"."l_shipdate",
              "t4"."l_commitdate",
              "t4"."l_receiptdate",
              "t4"."l_shipinstruct",
              "t4"."l_shipmode",
              "t4"."l_comment"
            FROM "lineitem" AS "t4"
            WHERE
              "t4"."l_partkey" = "t1"."ps_partkey"
              AND "t4"."l_suppkey" = "t1"."ps_suppkey"
              AND "t4"."l_shipdate" >= MAKE_DATE(1994, 1, 1)
              AND "t4"."l_shipdate" < MAKE_DATE(1995, 1, 1)
          ) AS "t8"
        ) * CAST(0.5 AS DOUBLE)
      )
  )
ORDER BY
  "t9"."s_name" ASC