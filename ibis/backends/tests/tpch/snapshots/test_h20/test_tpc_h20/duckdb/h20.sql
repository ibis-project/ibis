SELECT
  t0.s_name,
  t0.s_address
FROM (
  SELECT
    t1.s_name AS s_name,
    t1.s_address AS s_address
  FROM main.supplier AS t1
  JOIN main.nation AS t2
    ON t1.s_nationkey = t2.n_nationkey
  WHERE
    t2.n_name = 'CANADA'
    AND t1.s_suppkey IN (
      SELECT
        t3.ps_suppkey
      FROM main.partsupp AS t3
      WHERE
        t3.ps_partkey IN (
          SELECT
            t4.p_partkey
          FROM main.part AS t4
          WHERE
            t4.p_name LIKE 'forest%'
        )
        AND t3.ps_availqty > (
          SELECT
            SUM(t4.l_quantity) AS "Sum(l_quantity)"
          FROM main.lineitem AS t4
          WHERE
            t4.l_partkey = t3.ps_partkey
            AND t4.l_suppkey = t3.ps_suppkey
            AND t4.l_shipdate >= CAST('1994-01-01' AS DATE)
            AND t4.l_shipdate < CAST('1995-01-01' AS DATE)
        ) * CAST(0.5 AS REAL(53))
    )
) AS t0
ORDER BY
  t0.s_name ASC