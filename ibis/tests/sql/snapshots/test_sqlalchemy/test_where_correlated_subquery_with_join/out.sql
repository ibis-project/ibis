SELECT
  t0.p_partkey,
  t0.ps_supplycost
FROM (
  SELECT
    t1.p_partkey AS p_partkey,
    t2.ps_supplycost AS ps_supplycost
  FROM part AS t1
  JOIN partsupp AS t2
    ON t1.p_partkey = t2.ps_partkey
) AS t0
WHERE
  t0.ps_supplycost = (
    SELECT
      MIN(t1.ps_supplycost) AS "Min(ps_supplycost)"
    FROM (
      SELECT
        t2.ps_partkey AS ps_partkey,
        t2.ps_supplycost AS ps_supplycost
      FROM partsupp AS t2
      JOIN supplier AS t3
        ON t3.s_suppkey = t2.ps_suppkey
    ) AS t1
    WHERE
      t1.ps_partkey = t0.p_partkey
  )