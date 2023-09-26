WITH t0 AS (
  SELECT
    CAST(EXTRACT(year FROM t7.o_orderdate) AS SMALLINT) AS o_year,
    t5.l_extendedprice * (
      CAST(1 AS TINYINT) - t5.l_discount
    ) AS volume,
    t11.n_name AS nation,
    t10.r_name AS r_name,
    t7.o_orderdate AS o_orderdate,
    t4.p_type AS p_type
  FROM main.part AS t4
  JOIN main.lineitem AS t5
    ON t4.p_partkey = t5.l_partkey
  JOIN main.supplier AS t6
    ON t6.s_suppkey = t5.l_suppkey
  JOIN main.orders AS t7
    ON t5.l_orderkey = t7.o_orderkey
  JOIN main.customer AS t8
    ON t7.o_custkey = t8.c_custkey
  JOIN main.nation AS t9
    ON t8.c_nationkey = t9.n_nationkey
  JOIN main.region AS t10
    ON t9.n_regionkey = t10.r_regionkey
  JOIN main.nation AS t11
    ON t6.s_nationkey = t11.n_nationkey
), t1 AS (
  SELECT
    t0.o_year AS o_year,
    t0.volume AS volume,
    t0.nation AS nation,
    t0.r_name AS r_name,
    t0.o_orderdate AS o_orderdate,
    t0.p_type AS p_type
  FROM t0
  WHERE
    t0.r_name = 'AMERICA'
    AND t0.o_orderdate BETWEEN CAST('1995-01-01' AS DATE) AND CAST('1996-12-31' AS DATE)
    AND t0.p_type = 'ECONOMY ANODIZED STEEL'
), t2 AS (
  SELECT
    t1.o_year AS o_year,
    t1.volume AS volume,
    t1.nation AS nation,
    t1.r_name AS r_name,
    t1.o_orderdate AS o_orderdate,
    t1.p_type AS p_type,
    CASE WHEN (
      t1.nation = 'BRAZIL'
    ) THEN t1.volume ELSE CAST(0 AS TINYINT) END AS nation_volume
  FROM t1
)
SELECT
  t3.o_year,
  t3.mkt_share
FROM (
  SELECT
    t2.o_year AS o_year,
    SUM(t2.nation_volume) / SUM(t2.volume) AS mkt_share
  FROM t2
  GROUP BY
    1
) AS t3
ORDER BY
  t3.o_year ASC