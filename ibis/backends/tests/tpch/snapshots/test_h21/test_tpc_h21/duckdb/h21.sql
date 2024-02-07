SELECT
  "t14"."s_name",
  "t14"."numwait"
FROM (
  SELECT
    "t13"."s_name",
    COUNT(*) AS "numwait"
  FROM (
    SELECT
      "t10"."l1_orderkey",
      "t10"."o_orderstatus",
      "t10"."l_receiptdate",
      "t10"."l_commitdate",
      "t10"."l1_suppkey",
      "t10"."s_name",
      "t10"."n_name"
    FROM (
      SELECT
        "t5"."l_orderkey" AS "l1_orderkey",
        "t8"."o_orderstatus",
        "t5"."l_receiptdate",
        "t5"."l_commitdate",
        "t5"."l_suppkey" AS "l1_suppkey",
        "t4"."s_name",
        "t9"."n_name"
      FROM "supplier" AS "t4"
      INNER JOIN "lineitem" AS "t5"
        ON "t4"."s_suppkey" = "t5"."l_suppkey"
      INNER JOIN "orders" AS "t8"
        ON "t8"."o_orderkey" = "t5"."l_orderkey"
      INNER JOIN "nation" AS "t9"
        ON "t4"."s_nationkey" = "t9"."n_nationkey"
    ) AS "t10"
    WHERE
      "t10"."o_orderstatus" = 'F'
      AND "t10"."l_receiptdate" > "t10"."l_commitdate"
      AND "t10"."n_name" = 'SAUDI ARABIA'
      AND EXISTS(
        SELECT
          1
        FROM "lineitem" AS "t6"
        WHERE
          (
            "t6"."l_orderkey" = "t10"."l1_orderkey"
          )
          AND (
            "t6"."l_suppkey" <> "t10"."l1_suppkey"
          )
      )
      AND NOT (
        EXISTS(
          SELECT
            1
          FROM "lineitem" AS "t7"
          WHERE
            (
              (
                "t7"."l_orderkey" = "t10"."l1_orderkey"
              )
              AND (
                "t7"."l_suppkey" <> "t10"."l1_suppkey"
              )
            )
            AND (
              "t7"."l_receiptdate" > "t7"."l_commitdate"
            )
        )
      )
  ) AS "t13"
  GROUP BY
    1
) AS "t14"
ORDER BY
  "t14"."numwait" DESC,
  "t14"."s_name" ASC
LIMIT 100