WITH `t12` AS (
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
)
SELECT
  `t14`.`year`,
  `t14`.`CountStar()` AS `pre_count`,
  `t16`.`CountStar()` AS `post_count`
FROM `t12` AS `t14`
INNER JOIN `t12` AS `t16`
  ON `t14`.`year` = `t16`.`year`