SELECT max(t0.`f`) OVER (ORDER BY t0.`d` ASC) AS `foo`
FROM `alltypes` t0