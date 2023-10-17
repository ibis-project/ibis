SELECT
  SUM(t0.l_extendedprice * (
    1 - t0.l_discount
  )) AS revenue
FROM hive.ibis_sf1.lineitem AS t0
JOIN hive.ibis_sf1.part AS t1
  ON t1.p_partkey = t0.l_partkey
WHERE
  t1.p_brand = 'Brand#12'
  AND t1.p_container IN ('SM CASE', 'SM BOX', 'SM PACK', 'SM PKG')
  AND t0.l_quantity >= 1
  AND t0.l_quantity <= 11
  AND t1.p_size BETWEEN 1 AND 5
  AND t0.l_shipmode IN ('AIR', 'AIR REG')
  AND t0.l_shipinstruct = 'DELIVER IN PERSON'
  OR t1.p_brand = 'Brand#23'
  AND t1.p_container IN ('MED BAG', 'MED BOX', 'MED PKG', 'MED PACK')
  AND t0.l_quantity >= 10
  AND t0.l_quantity <= 20
  AND t1.p_size BETWEEN 1 AND 10
  AND t0.l_shipmode IN ('AIR', 'AIR REG')
  AND t0.l_shipinstruct = 'DELIVER IN PERSON'
  OR t1.p_brand = 'Brand#34'
  AND t1.p_container IN ('LG CASE', 'LG BOX', 'LG PACK', 'LG PKG')
  AND t0.l_quantity >= 20
  AND t0.l_quantity <= 30
  AND t1.p_size BETWEEN 1 AND 15
  AND t0.l_shipmode IN ('AIR', 'AIR REG')
  AND t0.l_shipinstruct = 'DELIVER IN PERSON'