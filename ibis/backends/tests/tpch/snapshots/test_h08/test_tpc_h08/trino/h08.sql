WITH t0 AS (
  SELECT
    CAST(EXTRACT(year FROM t6.o_orderdate) AS SMALLINT) AS o_year,
    t4.l_extendedprice * (
      1 - t4.l_discount
    ) AS volume,
    t10.n_name AS nation,
    t9.r_name AS r_name,
    t6.o_orderdate AS o_orderdate,
    t3.p_type AS p_type
  FROM hive.ibis_sf1.part AS t3
  JOIN hive.ibis_sf1.lineitem AS t4
    ON t3.p_partkey = t4.l_partkey
  JOIN hive.ibis_sf1.supplier AS t5
    ON t5.s_suppkey = t4.l_suppkey
  JOIN hive.ibis_sf1.orders AS t6
    ON t4.l_orderkey = t6.o_orderkey
  JOIN hive.ibis_sf1.customer AS t7
    ON t6.o_custkey = t7.c_custkey
  JOIN hive.ibis_sf1.nation AS t8
    ON t7.c_nationkey = t8.n_nationkey
  JOIN hive.ibis_sf1.region AS t9
    ON t8.n_regionkey = t9.r_regionkey
  JOIN hive.ibis_sf1.nation AS t10
    ON t5.s_nationkey = t10.n_nationkey
), t1 AS (
  SELECT
    t0.o_year AS o_year,
    t0.volume AS volume,
    t0.nation AS nation,
    t0.r_name AS r_name,
    t0.o_orderdate AS o_orderdate,
    t0.p_type AS p_type,
    CASE WHEN (
      t0.nation = 'BRAZIL'
    ) THEN t0.volume ELSE 0 END AS nation_volume
  FROM t0
  WHERE
    t0.r_name = 'AMERICA'
    AND t0.o_orderdate BETWEEN FROM_ISO8601_DATE('1995-01-01') AND FROM_ISO8601_DATE('1996-12-31')
    AND t0.p_type = 'ECONOMY ANODIZED STEEL'
)
SELECT
  t2.o_year,
  t2.mkt_share
FROM (
  SELECT
    t1.o_year AS o_year,
    SUM(t1.nation_volume) / SUM(t1.volume) AS mkt_share
  FROM t1
  GROUP BY
    1
) AS t2
ORDER BY
  t2.o_year ASC