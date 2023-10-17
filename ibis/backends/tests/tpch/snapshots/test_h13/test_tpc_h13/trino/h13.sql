WITH t0 AS (
  SELECT
    t2.c_custkey AS c_custkey,
    COUNT(t3.o_orderkey) AS c_count
  FROM hive.ibis_sf1.customer AS t2
  LEFT OUTER JOIN hive.ibis_sf1.orders AS t3
    ON t2.c_custkey = t3.o_custkey AND NOT t3.o_comment LIKE '%special%requests%'
  GROUP BY
    1
)
SELECT
  t1.c_count,
  t1.custdist
FROM (
  SELECT
    t0.c_count AS c_count,
    COUNT(*) AS custdist
  FROM t0
  GROUP BY
    1
) AS t1
ORDER BY
  t1.custdist DESC,
  t1.c_count DESC