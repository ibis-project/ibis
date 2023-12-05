WITH t0 AS (
  SELECT
    t2.ps_partkey AS ps_partkey,
    t2.ps_supplycost AS ps_supplycost
  FROM partsupp AS t2
  JOIN supplier AS t3
    ON t3.s_suppkey = t2.ps_suppkey
  WHERE
    t2.ps_partkey = (
      SELECT
        anon_1.p_partkey
      FROM (
        SELECT
          t1.p_partkey AS p_partkey,
          t2.ps_partkey AS ps_partkey,
          t2.ps_supplycost AS ps_supplycost,
          t2.ps_suppkey AS ps_suppkey
        FROM part AS t1
        JOIN partsupp AS t2
          ON t1.p_partkey = t2.ps_partkey
      ) AS anon_1
    )
)
SELECT
  t1.p_partkey,
  t2.ps_supplycost
FROM part AS t1
JOIN partsupp AS t2
  ON t1.p_partkey = t2.ps_partkey
WHERE
  t2.ps_supplycost = (
    SELECT
      MIN(t0.ps_supplycost) AS "Min(ps_supplycost)"
    FROM t0
  )