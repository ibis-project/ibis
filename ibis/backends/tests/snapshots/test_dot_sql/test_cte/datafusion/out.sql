WITH "foo" AS (
  SELECT
    *
  FROM "test_datafusion_temp_mem_t_for_cte" AS "t0"
)
SELECT
  COUNT(*) AS "x"
FROM "foo"