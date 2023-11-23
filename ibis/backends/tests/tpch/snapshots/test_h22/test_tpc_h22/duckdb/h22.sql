SELECT
  *
FROM (
  SELECT
    t7.cntrycode AS cntrycode,
    COUNT(*) AS numcust,
    SUM(t7.c_acctbal) AS totacctbal
  FROM (
    SELECT
      CASE
        WHEN CAST(0 AS TINYINT) >= 0
        THEN SUBSTRING(t6.c_phone, CAST(0 AS TINYINT) + 1, CAST(2 AS TINYINT))
        ELSE SUBSTRING(t6.c_phone, CAST(0 AS TINYINT), CAST(2 AS TINYINT))
      END AS cntrycode,
      t6.c_acctbal AS c_acctbal
    FROM (
      SELECT
        *
      FROM "customer" AS t0
      WHERE
        CASE
          WHEN CAST(0 AS TINYINT) >= 0
          THEN SUBSTRING(t0.c_phone, CAST(0 AS TINYINT) + 1, CAST(2 AS TINYINT))
          ELSE SUBSTRING(t0.c_phone, CAST(0 AS TINYINT), CAST(2 AS TINYINT))
        END IN ('13', '31', '23', '29', '30', '18', '17')
        AND (
          t0.c_acctbal > (
            SELECT
              AVG(t3.c_acctbal) AS "Mean(c_acctbal)"
            FROM (
              SELECT
                *
              FROM "customer" AS t0
              WHERE
                (
                  t0.c_acctbal > CAST(0.0 AS DOUBLE)
                )
                AND CASE
                  WHEN CAST(0 AS TINYINT) >= 0
                  THEN SUBSTRING(t0.c_phone, CAST(0 AS TINYINT) + 1, CAST(2 AS TINYINT))
                  ELSE SUBSTRING(t0.c_phone, CAST(0 AS TINYINT), CAST(2 AS TINYINT))
                END IN ('13', '31', '23', '29', '30', '18', '17')
            ) AS t3
          )
        )
        AND NOT EXISTS(
          (
            SELECT
              CAST(1 AS TINYINT) AS "1"
            FROM (
              SELECT
                *
              FROM "orders" AS t1
              WHERE
                (
                  t1.o_custkey = t0.c_custkey
                )
            ) AS t2
          )
        )
    ) AS t6
  ) AS t7
  GROUP BY
    1
) AS t8
ORDER BY
  t8.cntrycode ASC