SELECT
  *
FROM (
  SELECT
    "t14"."s_name",
    "t14"."s_address"
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
          "t12"."ps_suppkey"
        FROM (
          SELECT
            *
          FROM "partsupp" AS "t2"
          WHERE
            "t2"."ps_partkey" IN (
              SELECT
                "t7"."p_partkey"
              FROM (
                SELECT
                  *
                FROM "part" AS "t3"
                WHERE
                  "t3"."p_name" LIKE 'forest%'
              ) AS "t7"
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
                    AND "t4"."l_shipdate" >= MAKE_DATE(1994, 1, 1)
                    AND "t4"."l_shipdate" < MAKE_DATE(1995, 1, 1)
                ) AS "t8"
              ) * CAST(0.5 AS DOUBLE)
            )
        ) AS "t12"
      )
  ) AS "t14"
) AS "t15"
ORDER BY
  "t15"."s_name" ASC