SELECT
  SUM(t0.l_extendedprice * (
    CAST(1 AS TINYINT) - t0.l_discount
  )) AS revenue
FROM main.lineitem AS t0
JOIN main.part AS t1
  ON t1.p_partkey = t0.l_partkey
WHERE
  t1.p_brand = 'Brand#12'
  AND t1.p_container IN ('SM CASE', 'SM BOX', 'SM PACK', 'SM PKG')
  AND t0.l_quantity >= CAST(1 AS TINYINT)
  AND t0.l_quantity <= CAST(11 AS TINYINT)
  AND t1.p_size BETWEEN CAST(1 AS TINYINT) AND CAST(5 AS TINYINT)
  AND t0.l_shipmode IN ('AIR', 'AIR REG')
  AND t0.l_shipinstruct = 'DELIVER IN PERSON'
  OR t1.p_brand = 'Brand#23'
  AND t1.p_container IN ('MED BAG', 'MED BOX', 'MED PKG', 'MED PACK')
  AND t0.l_quantity >= CAST(10 AS TINYINT)
  AND t0.l_quantity <= CAST(20 AS TINYINT)
  AND t1.p_size BETWEEN CAST(1 AS TINYINT) AND CAST(10 AS TINYINT)
  AND t0.l_shipmode IN ('AIR', 'AIR REG')
  AND t0.l_shipinstruct = 'DELIVER IN PERSON'
  OR t1.p_brand = 'Brand#34'
  AND t1.p_container IN ('LG CASE', 'LG BOX', 'LG PACK', 'LG PKG')
  AND t0.l_quantity >= CAST(20 AS TINYINT)
  AND t0.l_quantity <= CAST(30 AS TINYINT)
  AND t1.p_size BETWEEN CAST(1 AS TINYINT) AND CAST(15 AS TINYINT)
  AND t0.l_shipmode IN ('AIR', 'AIR REG')
  AND t0.l_shipinstruct = 'DELIVER IN PERSON'