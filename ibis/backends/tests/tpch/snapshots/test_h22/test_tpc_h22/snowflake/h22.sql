SELECT
  "t6"."cntrycode" AS "cntrycode",
  "t6"."numcust" AS "numcust",
  "t6"."totacctbal" AS "totacctbal"
FROM (
  SELECT
    "t5"."cntrycode" AS "cntrycode",
    COUNT(*) AS "numcust",
    SUM("t5"."c_acctbal") AS "totacctbal"
  FROM (
    SELECT
      IFF(0 >= 0, SUBSTRING("t0"."C_PHONE", 0 + 1, 2), SUBSTRING("t0"."C_PHONE", 0, 2)) AS "cntrycode",
      "t0"."C_ACCTBAL" AS "c_acctbal"
    FROM "CUSTOMER" AS "t0"
    WHERE
      IFF(0 >= 0, SUBSTRING("t0"."C_PHONE", 0 + 1, 2), SUBSTRING("t0"."C_PHONE", 0, 2)) IN ('13', '31', '23', '29', '30', '18', '17')
      AND "t0"."C_ACCTBAL" > (
        SELECT
          AVG("t3"."c_acctbal") AS "Mean(c_acctbal)"
        FROM (
          SELECT
            "t0"."C_CUSTKEY" AS "c_custkey",
            "t0"."C_NAME" AS "c_name",
            "t0"."C_ADDRESS" AS "c_address",
            "t0"."C_NATIONKEY" AS "c_nationkey",
            "t0"."C_PHONE" AS "c_phone",
            "t0"."C_ACCTBAL" AS "c_acctbal",
            "t0"."C_MKTSEGMENT" AS "c_mktsegment",
            "t0"."C_COMMENT" AS "c_comment"
          FROM "CUSTOMER" AS "t0"
          WHERE
            "t0"."C_ACCTBAL" > 0.0
            AND IFF(0 >= 0, SUBSTRING("t0"."C_PHONE", 0 + 1, 2), SUBSTRING("t0"."C_PHONE", 0, 2)) IN ('13', '31', '23', '29', '30', '18', '17')
        ) AS "t3"
      )
      AND NOT (
        EXISTS(
          (
            SELECT
              1 AS "1"
            FROM "ORDERS" AS "t1"
            WHERE
              "t1"."O_CUSTKEY" = "t0"."C_CUSTKEY"
          )
        )
      )
  ) AS "t5"
  GROUP BY
    1
) AS "t6"
ORDER BY
  "t6"."cntrycode" ASC