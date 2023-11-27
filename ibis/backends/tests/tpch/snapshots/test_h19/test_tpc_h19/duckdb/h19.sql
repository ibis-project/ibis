SELECT
  SUM(t4.l_extendedprice * (
    CAST(1 AS TINYINT) - t4.l_discount
  )) AS revenue
FROM (
  SELECT
    *
  FROM (
    SELECT
      t0.l_orderkey AS l_orderkey,
      t0.l_partkey AS l_partkey,
      t0.l_suppkey AS l_suppkey,
      t0.l_linenumber AS l_linenumber,
      t0.l_quantity AS l_quantity,
      t0.l_extendedprice AS l_extendedprice,
      t0.l_discount AS l_discount,
      t0.l_tax AS l_tax,
      t0.l_returnflag AS l_returnflag,
      t0.l_linestatus AS l_linestatus,
      t0.l_shipdate AS l_shipdate,
      t0.l_commitdate AS l_commitdate,
      t0.l_receiptdate AS l_receiptdate,
      t0.l_shipinstruct AS l_shipinstruct,
      t0.l_shipmode AS l_shipmode,
      t0.l_comment AS l_comment,
      t1.p_partkey AS p_partkey,
      t1.p_name AS p_name,
      t1.p_mfgr AS p_mfgr,
      t1.p_brand AS p_brand,
      t1.p_type AS p_type,
      t1.p_size AS p_size,
      t1.p_container AS p_container,
      t1.p_retailprice AS p_retailprice,
      t1.p_comment AS p_comment
    FROM "lineitem" AS t0
    INNER JOIN "part" AS t1
      ON t1.p_partkey = t0.l_partkey
  ) AS t3
  WHERE
    (
      (
        (
          (
            (
              (
                (
                  (
                    (
                      t3.p_brand = 'Brand#12'
                    )
                    AND t3.p_container IN ('SM CASE', 'SM BOX', 'SM PACK', 'SM PKG')
                  )
                  AND (
                    t3.l_quantity >= CAST(1 AS TINYINT)
                  )
                )
                AND (
                  t3.l_quantity <= CAST(11 AS TINYINT)
                )
              )
              AND t3.p_size BETWEEN CAST(1 AS TINYINT) AND CAST(5 AS TINYINT)
            )
            AND t3.l_shipmode IN ('AIR', 'AIR REG')
          )
          AND (
            t3.l_shipinstruct = 'DELIVER IN PERSON'
          )
        )
        OR (
          (
            (
              (
                (
                  (
                    (
                      t3.p_brand = 'Brand#23'
                    )
                    AND t3.p_container IN ('MED BAG', 'MED BOX', 'MED PKG', 'MED PACK')
                  )
                  AND (
                    t3.l_quantity >= CAST(10 AS TINYINT)
                  )
                )
                AND (
                  t3.l_quantity <= CAST(20 AS TINYINT)
                )
              )
              AND t3.p_size BETWEEN CAST(1 AS TINYINT) AND CAST(10 AS TINYINT)
            )
            AND t3.l_shipmode IN ('AIR', 'AIR REG')
          )
          AND (
            t3.l_shipinstruct = 'DELIVER IN PERSON'
          )
        )
      )
      OR (
        (
          (
            (
              (
                (
                  (
                    t3.p_brand = 'Brand#34'
                  )
                  AND t3.p_container IN ('LG CASE', 'LG BOX', 'LG PACK', 'LG PKG')
                )
                AND (
                  t3.l_quantity >= CAST(20 AS TINYINT)
                )
              )
              AND (
                t3.l_quantity <= CAST(30 AS TINYINT)
              )
            )
            AND t3.p_size BETWEEN CAST(1 AS TINYINT) AND CAST(15 AS TINYINT)
          )
          AND t3.l_shipmode IN ('AIR', 'AIR REG')
        )
        AND (
          t3.l_shipinstruct = 'DELIVER IN PERSON'
        )
      )
    )
) AS t4