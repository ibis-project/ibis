SELECT
  (
    SUM(
      CASE
        WHEN t5.p_type LIKE 'PROMO%'
        THEN t5.l_extendedprice * (
          CAST(1 AS TINYINT) - t5.l_discount
        )
        ELSE CAST(0 AS TINYINT)
      END
    ) * CAST(100 AS TINYINT)
  ) / SUM(t5.l_extendedprice * (
    CAST(1 AS TINYINT) - t5.l_discount
  )) AS promo_revenue
FROM (
  SELECT
    t4.l_orderkey AS l_orderkey,
    t4.l_partkey AS l_partkey,
    t4.l_suppkey AS l_suppkey,
    t4.l_linenumber AS l_linenumber,
    t4.l_quantity AS l_quantity,
    t4.l_extendedprice AS l_extendedprice,
    t4.l_discount AS l_discount,
    t4.l_tax AS l_tax,
    t4.l_returnflag AS l_returnflag,
    t4.l_linestatus AS l_linestatus,
    t4.l_shipdate AS l_shipdate,
    t4.l_commitdate AS l_commitdate,
    t4.l_receiptdate AS l_receiptdate,
    t4.l_shipinstruct AS l_shipinstruct,
    t4.l_shipmode AS l_shipmode,
    t4.l_comment AS l_comment,
    t4.p_partkey AS p_partkey,
    t4.p_name AS p_name,
    t4.p_mfgr AS p_mfgr,
    t4.p_brand AS p_brand,
    t4.p_type AS p_type,
    t4.p_size AS p_size,
    t4.p_container AS p_container,
    t4.p_retailprice AS p_retailprice,
    t4.p_comment AS p_comment
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
      t2.p_partkey AS p_partkey,
      t2.p_name AS p_name,
      t2.p_mfgr AS p_mfgr,
      t2.p_brand AS p_brand,
      t2.p_type AS p_type,
      t2.p_size AS p_size,
      t2.p_container AS p_container,
      t2.p_retailprice AS p_retailprice,
      t2.p_comment AS p_comment
    FROM lineitem AS t0
    INNER JOIN part AS t2
      ON t0.l_partkey = t2.p_partkey
  ) AS t4
  WHERE
    t4.l_shipdate >= MAKE_DATE(1995, 9, 1) AND t4.l_shipdate < MAKE_DATE(1995, 10, 1)
) AS t5