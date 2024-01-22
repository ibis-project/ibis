WITH `t8` AS (
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
)
SELECT
  `t12`.`year`,
  `t12`.`CountStar()` AS `pre_count`,
  `t17`.`CountStar()` AS `post_count`,
  `t17`.`CountStar()` / CAST(`t12`.`CountStar()` AS DOUBLE) AS `fraction`
FROM (
  SELECT
    EXTRACT(year FROM `t9`.`odate`) AS `year`,
    COUNT(*) AS `CountStar()`
  FROM `t8` AS `t9`
  GROUP BY
    1
) AS `t12`
INNER JOIN (
  SELECT
    EXTRACT(year FROM `t15`.`odate`) AS `year`,
    COUNT(*) AS `CountStar()`
  FROM (
    SELECT
      `t9`.`c_custkey`,
      `t9`.`c_name`,
      `t9`.`c_address`,
      `t9`.`c_nationkey`,
      `t9`.`c_phone`,
      `t9`.`c_acctbal`,
      `t9`.`c_mktsegment`,
      `t9`.`c_comment`,
      `t9`.`region`,
      `t9`.`o_totalprice`,
      `t9`.`odate`
    FROM `t8` AS `t9`
    WHERE
      `t9`.`o_totalprice` > (
        SELECT
          AVG(`t13`.`o_totalprice`) AS `Mean(o_totalprice)`
        FROM (
          SELECT
            `t10`.`c_custkey`,
            `t10`.`c_name`,
            `t10`.`c_address`,
            `t10`.`c_nationkey`,
            `t10`.`c_phone`,
            `t10`.`c_acctbal`,
            `t10`.`c_mktsegment`,
            `t10`.`c_comment`,
            `t10`.`region`,
            `t10`.`o_totalprice`,
            `t10`.`odate`
          FROM `t8` AS `t10`
          WHERE
            `t10`.`region` = `t9`.`region`
        ) AS `t13`
      )
  ) AS `t15`
  GROUP BY
    1
) AS `t17`
  ON `t12`.`year` = `t17`.`year`