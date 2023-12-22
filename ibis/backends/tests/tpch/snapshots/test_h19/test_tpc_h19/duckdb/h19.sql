SELECT
  SUM(t6.l_extendedprice * (
    CAST(1 AS TINYINT) - t6.l_discount
  )) AS revenue
FROM (
  SELECT
    t5.l_orderkey,
    t5.l_partkey,
    t5.l_suppkey,
    t5.l_linenumber,
    t5.l_quantity,
    t5.l_extendedprice,
    t5.l_discount,
    t5.l_tax,
    t5.l_returnflag,
    t5.l_linestatus,
    t5.l_shipdate,
    t5.l_commitdate,
    t5.l_receiptdate,
    t5.l_shipinstruct,
    t5.l_shipmode,
    t5.l_comment,
    t5.p_partkey,
    t5.p_name,
    t5.p_mfgr,
    t5.p_brand,
    t5.p_type,
    t5.p_size,
    t5.p_container,
    t5.p_retailprice,
    t5.p_comment
  FROM (
    SELECT
      t2.l_orderkey,
      t2.l_partkey,
      t2.l_suppkey,
      t2.l_linenumber,
      t2.l_quantity,
      t2.l_extendedprice,
      t2.l_discount,
      t2.l_tax,
      t2.l_returnflag,
      t2.l_linestatus,
      t2.l_shipdate,
      t2.l_commitdate,
      t2.l_receiptdate,
      t2.l_shipinstruct,
      t2.l_shipmode,
      t2.l_comment,
      t3.p_partkey,
      t3.p_name,
      t3.p_mfgr,
      t3.p_brand,
      t3.p_type,
      t3.p_size,
      t3.p_container,
      t3.p_retailprice,
      t3.p_comment
    FROM lineitem AS t2
    INNER JOIN part AS t3
      ON t3.p_partkey = t2.l_partkey
  ) AS t5
  WHERE
    (
      (
        (
          (
            (
              (
                (
                  (
                    t5.p_brand = 'Brand#12'
                  )
                  AND t5.p_container IN ('SM CASE', 'SM BOX', 'SM PACK', 'SM PKG')
                )
                AND (
                  t5.l_quantity >= CAST(1 AS TINYINT)
                )
              )
              AND (
                t5.l_quantity <= CAST(11 AS TINYINT)
              )
            )
            AND t5.p_size BETWEEN CAST(1 AS TINYINT) AND CAST(5 AS TINYINT)
          )
          AND t5.l_shipmode IN ('AIR', 'AIR REG')
        )
        AND (
          t5.l_shipinstruct = 'DELIVER IN PERSON'
        )
      )
      OR (
        (
          (
            (
              (
                (
                  (
                    t5.p_brand = 'Brand#23'
                  )
                  AND t5.p_container IN ('MED BAG', 'MED BOX', 'MED PKG', 'MED PACK')
                )
                AND (
                  t5.l_quantity >= CAST(10 AS TINYINT)
                )
              )
              AND (
                t5.l_quantity <= CAST(20 AS TINYINT)
              )
            )
            AND t5.p_size BETWEEN CAST(1 AS TINYINT) AND CAST(10 AS TINYINT)
          )
          AND t5.l_shipmode IN ('AIR', 'AIR REG')
        )
        AND (
          t5.l_shipinstruct = 'DELIVER IN PERSON'
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
                  t5.p_brand = 'Brand#34'
                )
                AND t5.p_container IN ('LG CASE', 'LG BOX', 'LG PACK', 'LG PKG')
              )
              AND (
                t5.l_quantity >= CAST(20 AS TINYINT)
              )
            )
            AND (
              t5.l_quantity <= CAST(30 AS TINYINT)
            )
          )
          AND t5.p_size BETWEEN CAST(1 AS TINYINT) AND CAST(15 AS TINYINT)
        )
        AND t5.l_shipmode IN ('AIR', 'AIR REG')
      )
      AND (
        t5.l_shipinstruct = 'DELIVER IN PERSON'
      )
    )
) AS t6