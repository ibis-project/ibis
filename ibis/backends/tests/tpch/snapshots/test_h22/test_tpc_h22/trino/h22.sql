SELECT
  *
FROM (
  SELECT
    "t7"."cntrycode",
    COUNT(*) AS "numcust",
    SUM("t7"."c_acctbal") AS "totacctbal"
  FROM (
    SELECT
      SUBSTRING("t6"."c_phone", IF((
        0 + 1
      ) >= 1, 0 + 1, 0 + 1 + LENGTH("t6"."c_phone")), 2) AS "cntrycode",
      "t6"."c_acctbal"
    FROM (
      SELECT
        *
      FROM (
        SELECT
          "t0"."c_custkey",
          "t0"."c_name",
          "t0"."c_address",
          "t0"."c_nationkey",
          "t0"."c_phone",
          CAST("t0"."c_acctbal" AS DECIMAL(15, 2)) AS "c_acctbal",
          "t0"."c_mktsegment",
          "t0"."c_comment"
        FROM "hive"."ibis_sf1"."customer" AS "t0"
      ) AS "t2"
      WHERE
        SUBSTRING("t2"."c_phone", IF((
          0 + 1
        ) >= 1, 0 + 1, 0 + 1 + LENGTH("t2"."c_phone")), 2) IN ('13', '31', '23', '29', '30', '18', '17')
        AND "t2"."c_acctbal" > (
          SELECT
            AVG("t3"."c_acctbal") AS "Mean(c_acctbal)"
          FROM (
            SELECT
              "t0"."c_custkey",
              "t0"."c_name",
              "t0"."c_address",
              "t0"."c_nationkey",
              "t0"."c_phone",
              CAST("t0"."c_acctbal" AS DECIMAL(15, 2)) AS "c_acctbal",
              "t0"."c_mktsegment",
              "t0"."c_comment"
            FROM "hive"."ibis_sf1"."customer" AS "t0"
            WHERE
              CAST("t0"."c_acctbal" AS DECIMAL(15, 2)) > CAST(0.0 AS DOUBLE)
              AND SUBSTRING("t0"."c_phone", IF((
                0 + 1
              ) >= 1, 0 + 1, 0 + 1 + LENGTH("t0"."c_phone")), 2) IN ('13', '31', '23', '29', '30', '18', '17')
          ) AS "t3"
        )
        AND NOT (
          EXISTS(
            SELECT
              1
            FROM "hive"."ibis_sf1"."orders" AS "t1"
            WHERE
              "t1"."o_custkey" = "t2"."c_custkey"
          )
        )
    ) AS "t6"
  ) AS "t7"
  GROUP BY
    1
) AS "t8"
ORDER BY
  "t8"."cntrycode" ASC