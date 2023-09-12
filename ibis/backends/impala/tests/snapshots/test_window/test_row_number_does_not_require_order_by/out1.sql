SELECT t0.*, (row_number() OVER (PARTITION BY t0.`g`) - 1) AS `foo`
FROM `alltypes` t0