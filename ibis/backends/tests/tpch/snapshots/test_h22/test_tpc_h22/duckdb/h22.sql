SELECT
  "t6"."cntrycode",
  "t6"."numcust",
  "t6"."totacctbal"
FROM (
  SELECT
    "t5"."cntrycode",
    COUNT(*) AS "numcust",
    SUM("t5"."c_acctbal") AS "totacctbal"
  FROM (
    SELECT
      CASE
        WHEN (
          CAST(0 AS TINYINT) + 1
        ) >= 1
        THEN SUBSTRING("t0"."c_phone", CAST(0 AS TINYINT) + 1, CAST(2 AS TINYINT))
        ELSE SUBSTRING("t0"."c_phone", CAST(0 AS TINYINT) + 1 + LENGTH("t0"."c_phone"), CAST(2 AS TINYINT))
      END AS "cntrycode",
      "t0"."c_acctbal"
    FROM "customer" AS "t0"
    WHERE
      CASE
        WHEN (
          CAST(0 AS TINYINT) + 1
        ) >= 1
        THEN SUBSTRING("t0"."c_phone", CAST(0 AS TINYINT) + 1, CAST(2 AS TINYINT))
        ELSE SUBSTRING("t0"."c_phone", CAST(0 AS TINYINT) + 1 + LENGTH("t0"."c_phone"), CAST(2 AS TINYINT))
      END IN ('13', '31', '23', '29', '30', '18', '17')
      AND "t0"."c_acctbal" > (
        SELECT
          AVG("t3"."c_acctbal") AS "Mean(c_acctbal)"
        FROM (
          SELECT
            "t0"."c_custkey",
            "t0"."c_name",
            "t0"."c_address",
            "t0"."c_nationkey",
            "t0"."c_phone",
            "t0"."c_acctbal",
            "t0"."c_mktsegment",
            "t0"."c_comment"
          FROM "customer" AS "t0"
          WHERE
            "t0"."c_acctbal" > CAST(0.0 AS DOUBLE)
            AND CASE
              WHEN (
                CAST(0 AS TINYINT) + 1
              ) >= 1
              THEN SUBSTRING("t0"."c_phone", CAST(0 AS TINYINT) + 1, CAST(2 AS TINYINT))
              ELSE SUBSTRING("t0"."c_phone", CAST(0 AS TINYINT) + 1 + LENGTH("t0"."c_phone"), CAST(2 AS TINYINT))
            END IN ('13', '31', '23', '29', '30', '18', '17')
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
  GROUP BY
    1
) AS "t6"
ORDER BY
  "t6"."cntrycode" ASC