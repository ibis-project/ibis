SELECT
  t5.c_name AS c_name,
  t5.r_name AS r_name,
  t5.n_name AS n_name
FROM (
  SELECT
    t0.c_custkey AS c_custkey,
    t0.c_name AS c_name,
    t0.c_address AS c_address,
    t0.c_nationkey AS c_nationkey,
    t0.c_phone AS c_phone,
    t0.c_acctbal AS c_acctbal,
    t0.c_mktsegment AS c_mktsegment,
    t0.c_comment AS c_comment,
    t1.n_name AS n_name,
    t2.r_name AS r_name
  FROM tpch_customer AS t0
  INNER JOIN tpch_nation AS t1
    ON t0.c_nationkey = t1.n_nationkey
  INNER JOIN tpch_region AS t2
    ON t1.n_regionkey = t2.r_regionkey
) AS t5
SEMI JOIN (
  SELECT
    *
  FROM (
    SELECT
      t5.n_name AS n_name,
      SUM(CAST(t5.c_acctbal AS DOUBLE)) AS "Sum(Cast(c_acctbal, float64))"
    FROM (
      SELECT
        t0.c_custkey AS c_custkey,
        t0.c_name AS c_name,
        t0.c_address AS c_address,
        t0.c_nationkey AS c_nationkey,
        t0.c_phone AS c_phone,
        t0.c_acctbal AS c_acctbal,
        t0.c_mktsegment AS c_mktsegment,
        t0.c_comment AS c_comment,
        t1.n_name AS n_name,
        t2.r_name AS r_name
      FROM tpch_customer AS t0
      INNER JOIN tpch_nation AS t1
        ON t0.c_nationkey = t1.n_nationkey
      INNER JOIN tpch_region AS t2
        ON t1.n_regionkey = t2.r_regionkey
    ) AS t5
    GROUP BY
      1
  ) AS t6
  ORDER BY
    t6."Sum(Cast(c_acctbal, float64))" DESC
  LIMIT 10
) AS t8
  ON t5.n_name = t8.n_name