SELECT t0.`f`, (row_number() OVER (ORDER BY t0.`f` DESC) - 1) AS `revrank`
FROM `alltypes` t0