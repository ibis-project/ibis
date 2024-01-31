WITH "foo" AS (
  SELECT
    *
  FROM "test_oracle_temp_mem_t_for_cte" "t0"
)
SELECT
  COUNT(*) AS "x"
FROM "foo"