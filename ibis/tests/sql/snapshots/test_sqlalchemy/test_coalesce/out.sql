SELECT
  COALESCE(
    CASE WHEN (
      t0.double_col > 30
    ) THEN t0.double_col ELSE NULL END,
    NULL,
    t0.float_col
  ) AS tmp
FROM functional_alltypes AS t0