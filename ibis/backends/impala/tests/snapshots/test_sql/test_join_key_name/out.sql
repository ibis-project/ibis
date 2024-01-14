SELECT
  `t14`.`year`,
  `t14`.`CountStar()` AS `pre_count`,
  `t19`.`CountStar()` AS `post_count`,
  `t19`.`CountStar()` / CAST(`t14`.`CountStar()` AS DOUBLE) AS `fraction`
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
) AS `t14`
INNER JOIN (
  SELECT
    EXTRACT(year FROM `t17`.`odate`) AS `year`,
    COUNT(*) AS `CountStar()`
  FROM (
    SELECT
      `t11`.`c_custkey`,
      `t11`.`c_name`,
      `t11`.`c_address`,
      `t11`.`c_nationkey`,
      `t11`.`c_phone`,
      `t11`.`c_acctbal`,
      `t11`.`c_mktsegment`,
      `t11`.`c_comment`,
      `t11`.`region`,
      `t11`.`o_totalprice`,
      `t11`.`odate`
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
    WHERE
      `t11`.`o_totalprice` > (
        SELECT
          AVG(`t15`.`o_totalprice`) AS `Mean(o_totalprice)`
        FROM (
          SELECT
            `t12`.`c_custkey`,
            `t12`.`c_name`,
            `t12`.`c_address`,
            `t12`.`c_nationkey`,
            `t12`.`c_phone`,
            `t12`.`c_acctbal`,
            `t12`.`c_mktsegment`,
            `t12`.`c_comment`,
            `t12`.`region`,
            `t12`.`o_totalprice`,
            `t12`.`odate`
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
          ) AS `t12`
          WHERE
            `t12`.`region` = `t11`.`region`
        ) AS `t15`
      )
  ) AS `t17`
  GROUP BY
    1
) AS `t19`
  ON `t14`.`year` = `t19`.`year`