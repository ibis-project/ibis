SELECT
  *
FROM (
  SELECT
    t0.p_partkey AS p_partkey,
    t1.ps_supplycost AS ps_supplycost
  FROM part AS t0
  INNER JOIN partsupp AS t1
    ON t0.p_partkey = t1.ps_partkey
) AS t5
WHERE
  (
    t5.ps_supplycost = (
      SELECT
        MIN(t7.ps_supplycost) AS "Min(ps_supplycost)"
      FROM (
        SELECT
          *
        FROM (
          SELECT
            t1.ps_partkey AS ps_partkey,
            t1.ps_supplycost AS ps_supplycost
          FROM partsupp AS t1
          INNER JOIN supplier AS t2
            ON t2.s_suppkey = t1.ps_suppkey
        ) AS t6
        WHERE
          (
            t6.ps_partkey = t5.p_partkey
          )
      ) AS t7
    )
  )