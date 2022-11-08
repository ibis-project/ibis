SELECT `f`, (row_number() OVER (ORDER BY `f` DESC) - 1) AS `revrank`
FROM alltypes