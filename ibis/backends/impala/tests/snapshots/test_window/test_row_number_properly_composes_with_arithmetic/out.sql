SELECT t0.*, (row_number() OVER (ORDER BY t0.`f` ASC) - 1) / 2 AS `new`
FROM `alltypes` t0