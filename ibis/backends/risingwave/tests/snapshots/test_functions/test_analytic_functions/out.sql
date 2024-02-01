SELECT
  RANK() OVER (ORDER BY t0.double_col ASC) - 1 AS rank,
  DENSE_RANK() OVER (ORDER BY t0.double_col ASC) - 1 AS dense_rank,
  CUME_DIST() OVER (ORDER BY t0.double_col ASC) AS cume_dist,
  NTILE(7) OVER (ORDER BY t0.double_col ASC) - 1 AS ntile,
  PERCENT_RANK() OVER (ORDER BY t0.double_col ASC) AS percent_rank
FROM functional_alltypes AS t0