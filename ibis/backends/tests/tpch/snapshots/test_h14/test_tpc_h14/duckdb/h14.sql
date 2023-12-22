SELECT
  (
    SUM(
      CASE
        WHEN t6.p_type LIKE 'PROMO%'
        THEN t6.l_extendedprice * (
          CAST(1 AS TINYINT) - t6.l_discount
        )
        ELSE CAST(0 AS TINYINT)
      END
    ) * CAST(100 AS TINYINT)
  ) / SUM(t6.l_extendedprice * (
    CAST(1 AS TINYINT) - t6.l_discount
  )) AS promo_revenue
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
      ON t2.l_partkey = t3.p_partkey
  ) AS t5
  WHERE
    t5.l_shipdate >= MAKE_DATE(1995, 9, 1) AND t5.l_shipdate < MAKE_DATE(1995, 10, 1)
) AS t6