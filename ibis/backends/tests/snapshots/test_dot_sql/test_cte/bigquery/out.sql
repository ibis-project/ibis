WITH `foo` AS (
  SELECT
    *
  FROM `ibis-gbq`.`ibis_gbq_testing`.`test_bigquery_temp_mem_t_for_cte` AS `t0`
)
SELECT
  COUNT(*) AS `x`
FROM `foo`