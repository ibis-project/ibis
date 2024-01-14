SELECT
  `t13`.`year`,
  `t13`.`CountStar()` AS `pre_count`,
  `t15`.`CountStar()` AS `post_count`
FROM (
  SELECT
    EXTRACT(year FROM `t11`.`odate`) AS `year`,
    COUNT(*) AS `CountStar()`
  FROM (
    SELECT
      `t6`.`c_custkey`,
      `t6`.`c_name`,
      `t6`.`c_address`,
      `t6`.`c_nationkey`,
      `t6`.`c_phone`,
      `t6`.`c_acctbal`,
      `t6`.`c_mktsegment`,
      `t6`.`c_comment`,
      `t4`.`r_name` AS `region`,
      `t7`.`o_totalprice`,
      CAST(`t7`.`o_orderdate` AS TIMESTAMP) AS `odate`
    FROM `tpch_region` AS `t4`
    INNER JOIN `tpch_nation` AS `t5`
      ON `t4`.`r_regionkey` = `t5`.`n_regionkey`
    INNER JOIN `tpch_customer` AS `t6`
      ON `t6`.`c_nationkey` = `t5`.`n_nationkey`
    INNER JOIN `tpch_orders` AS `t7`
      ON `t7`.`o_custkey` = `t6`.`c_custkey`
  ) AS `t11`
  GROUP BY
    1
) AS `t13`
INNER JOIN (
  SELECT
    EXTRACT(year FROM `t11`.`odate`) AS `year`,
    COUNT(*) AS `CountStar()`
  FROM (
    SELECT
      `t6`.`c_custkey`,
      `t6`.`c_name`,
      `t6`.`c_address`,
      `t6`.`c_nationkey`,
      `t6`.`c_phone`,
      `t6`.`c_acctbal`,
      `t6`.`c_mktsegment`,
      `t6`.`c_comment`,
      `t4`.`r_name` AS `region`,
      `t7`.`o_totalprice`,
      CAST(`t7`.`o_orderdate` AS TIMESTAMP) AS `odate`
    FROM `tpch_region` AS `t4`
    INNER JOIN `tpch_nation` AS `t5`
      ON `t4`.`r_regionkey` = `t5`.`n_regionkey`
    INNER JOIN `tpch_customer` AS `t6`
      ON `t6`.`c_nationkey` = `t5`.`n_nationkey`
    INNER JOIN `tpch_orders` AS `t7`
      ON `t7`.`o_custkey` = `t6`.`c_custkey`
  ) AS `t11`
  GROUP BY
    1
) AS `t15`
  ON `t13`.`year` = `t15`.`year`