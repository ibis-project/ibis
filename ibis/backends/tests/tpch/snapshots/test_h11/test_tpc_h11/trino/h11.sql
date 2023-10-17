WITH t0 AS (
  SELECT
    t2.ps_partkey AS ps_partkey,
    SUM(t2.ps_supplycost * t2.ps_availqty) AS value
  FROM hive.ibis_sf1.partsupp AS t2
  JOIN hive.ibis_sf1.supplier AS t3
    ON t2.ps_suppkey = t3.s_suppkey
  JOIN hive.ibis_sf1.nation AS t4
    ON t4.n_nationkey = t3.s_nationkey
  WHERE
    t4.n_name = 'GERMANY'
  GROUP BY
    1
)
SELECT
  t1.ps_partkey,
  t1.value
FROM (
  SELECT
    t0.ps_partkey AS ps_partkey,
    t0.value AS value
  FROM t0
  WHERE
    t0.value > (
      SELECT
        anon_1.total
      FROM (
        SELECT
          SUM(t2.ps_supplycost * t2.ps_availqty) AS total
        FROM hive.ibis_sf1.partsupp AS t2
        JOIN hive.ibis_sf1.supplier AS t3
          ON t2.ps_suppkey = t3.s_suppkey
        JOIN hive.ibis_sf1.nation AS t4
          ON t4.n_nationkey = t3.s_nationkey
        WHERE
          t4.n_name = 'GERMANY'
      ) AS anon_1
    ) * 0.0001
) AS t1
ORDER BY
  t1.value DESC