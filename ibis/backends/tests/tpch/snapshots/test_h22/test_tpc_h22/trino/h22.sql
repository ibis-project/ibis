WITH t0 AS (
  SELECT
    CASE
      WHEN (
        0 + 1 >= 1
      )
      THEN SUBSTR(t2.c_phone, 0 + 1, 2)
      ELSE SUBSTR(t2.c_phone, 0 + 1 + LENGTH(t2.c_phone), 2)
    END AS cntrycode,
    t2.c_acctbal AS c_acctbal
  FROM hive.ibis_sf1.customer AS t2
  WHERE
    CASE
      WHEN (
        0 + 1 >= 1
      )
      THEN SUBSTR(t2.c_phone, 0 + 1, 2)
      ELSE SUBSTR(t2.c_phone, 0 + 1 + LENGTH(t2.c_phone), 2)
    END IN ('13', '31', '23', '29', '30', '18', '17')
    AND t2.c_acctbal > (
      SELECT
        anon_1.avg_bal
      FROM (
        SELECT
          AVG(t2.c_acctbal) AS avg_bal
        FROM hive.ibis_sf1.customer AS t2
        WHERE
          t2.c_acctbal > 0.0
          AND CASE
            WHEN (
              0 + 1 >= 1
            )
            THEN SUBSTR(t2.c_phone, 0 + 1, 2)
            ELSE SUBSTR(t2.c_phone, 0 + 1 + LENGTH(t2.c_phone), 2)
          END IN ('13', '31', '23', '29', '30', '18', '17')
      ) AS anon_1
    )
    AND NOT (
      EXISTS(
        SELECT
          1 AS anon_2
        FROM hive.ibis_sf1.orders AS t3
        WHERE
          t3.o_custkey = t2.c_custkey
      )
    )
)
SELECT
  t1.cntrycode,
  t1.numcust,
  t1.totacctbal
FROM (
  SELECT
    t0.cntrycode AS cntrycode,
    COUNT(*) AS numcust,
    SUM(t0.c_acctbal) AS totacctbal
  FROM t0
  GROUP BY
    1
) AS t1
ORDER BY
  t1.cntrycode ASC