SELECT t0.`g`, (rank() OVER (ORDER BY t0.`f` ASC) - 1) AS `minr`,
       (dense_rank() OVER (ORDER BY t0.`f` ASC) - 1) AS `denser`
FROM `alltypes` t0