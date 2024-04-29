SELECT
  *
FROM (
  SELECT
    "t6"."cntrycode",
    COUNT(*) AS "numcust",
    SUM("t6"."c_acctbal") AS "totacctbal"
  FROM (
    SELECT
      SUBSTRING(
        "t5"."c_phone",
        CASE
          WHEN (
            CAST(0 AS TINYINT) + 1
          ) >= 1
          THEN CAST(0 AS TINYINT) + 1
          ELSE CAST(0 AS TINYINT) + 1 + LENGTH("t5"."c_phone")
        END,
        CAST(2 AS TINYINT)
      ) AS "cntrycode",
      "t5"."c_acctbal"
    FROM (
      SELECT
        *
      FROM "customer" AS "t0"
      WHERE
        SUBSTRING(
          "t0"."c_phone",
          CASE
            WHEN (
              CAST(0 AS TINYINT) + 1
            ) >= 1
            THEN CAST(0 AS TINYINT) + 1
            ELSE CAST(0 AS TINYINT) + 1 + LENGTH("t0"."c_phone")
          END,
          CAST(2 AS TINYINT)
        ) IN ('13', '31', '23', '29', '30', '18', '17')
        AND "t0"."c_acctbal" > (
          SELECT
            AVG("t3"."c_acctbal") AS "Mean(c_acctbal)"
          FROM (
            SELECT
              *
            FROM "customer" AS "t0"
            WHERE
              "t0"."c_acctbal" > CAST(0.0 AS DOUBLE)
              AND SUBSTRING(
                "t0"."c_phone",
                CASE
                  WHEN (
                    CAST(0 AS TINYINT) + 1
                  ) >= 1
                  THEN CAST(0 AS TINYINT) + 1
                  ELSE CAST(0 AS TINYINT) + 1 + LENGTH("t0"."c_phone")
                END,
                CAST(2 AS TINYINT)
              ) IN ('13', '31', '23', '29', '30', '18', '17')
          ) AS "t3"
        )
        AND NOT (
          EXISTS(
            SELECT
              1
            FROM "orders" AS "t1"
            WHERE
              "t1"."o_custkey" = "t0"."c_custkey"
          )
        )
    ) AS "t5"
  ) AS "t6"
  GROUP BY
    1
) AS "t7"
ORDER BY
  "t7"."cntrycode" ASC