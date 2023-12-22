SELECT
  t9.p_partkey,
  t9.ps_supplycost
FROM (
  SELECT
    t3.p_partkey,
    t4.ps_supplycost
  FROM part AS t3
  INNER JOIN partsupp AS t4
    ON t3.p_partkey = t4.ps_partkey
) AS t9
WHERE
  t9.ps_supplycost = (
    SELECT
      MIN(t11.ps_supplycost) AS "Min(ps_supplycost)"
    FROM (
      SELECT
        t10.ps_partkey,
        t10.ps_supplycost
      FROM (
        SELECT
          t5.ps_partkey,
          t5.ps_supplycost
        FROM partsupp AS t5
        INNER JOIN supplier AS t6
          ON t6.s_suppkey = t5.ps_suppkey
      ) AS t10
      WHERE
        t10.ps_partkey = t9.p_partkey
    ) AS t11
  )