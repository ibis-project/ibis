WITH t1 AS (
  SELECT
    t2.p_partkey AS p_partkey,
    t3.ps_supplycost AS ps_supplycost
  FROM part AS t2
  JOIN partsupp AS t3
    ON t2.p_partkey = t3.ps_partkey
), t0 AS (
  SELECT
    t2.ps_partkey AS ps_partkey,
    t2.ps_supplycost AS ps_supplycost
  FROM partsupp AS t2
  JOIN supplier AS t3
    ON t3.s_suppkey = t2.ps_suppkey
)
SELECT
  t1.p_partkey,
  t1.ps_supplycost
FROM t1
WHERE
  t1.ps_supplycost = (
    SELECT
      MIN(t0.ps_supplycost) AS "Min(ps_supplycost)"
    FROM t0
    WHERE
      t0.ps_partkey = t1.p_partkey
  )