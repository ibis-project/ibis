SELECT
  "t13"."s_name",
  "t13"."s_address"
FROM (
  SELECT
    *
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
        "t11"."ps_suppkey"
      FROM (
        SELECT
          *
        FROM "partsupp" AS "t2"
        WHERE
          "t2"."ps_partkey" IN (
            SELECT
              "t3"."p_partkey"
            FROM "part" AS "t3"
            WHERE
              "t3"."p_name" LIKE 'forest%'
          )
          AND "t2"."ps_availqty" > (
            (
              SELECT
                SUM("t8"."l_quantity") AS "Sum(l_quantity)"
              FROM (
                SELECT
                  *
                FROM "lineitem" AS "t4"
                WHERE
                  "t4"."l_partkey" = "t2"."ps_partkey"
                  AND "t4"."l_suppkey" = "t2"."ps_suppkey"
                  AND "t4"."l_shipdate" >= DATE_TRUNC('DAY', '1994-01-01')
                  AND "t4"."l_shipdate" < DATE_TRUNC('DAY', '1995-01-01')
              ) AS "t8"
            ) * 0.5
          )
      ) AS "t11"
    )
) AS "t13"
ORDER BY
  "t13"."s_name" ASC