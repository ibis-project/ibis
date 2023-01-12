SELECT t0.*,
       (row_number() OVER (PARTITION BY t0.`g` ORDER BY t0.`f`) - 1) AS `foo`
FROM alltypes t0