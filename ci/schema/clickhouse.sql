-- NB: The paths in this file are all relative to /var/lib/clickhouse/user_files

CREATE OR REPLACE TABLE diamonds ENGINE = Memory AS
SELECT * FROM file('ibis/diamonds.parquet', 'Parquet');

CREATE OR REPLACE TABLE batting ENGINE = Memory AS
SELECT * FROM file('ibis/batting.parquet', 'Parquet');

CREATE OR REPLACE TABLE awards_players ENGINE = Memory AS
SELECT * FROM file('ibis/awards_players.parquet', 'Parquet');

CREATE OR REPLACE TABLE functional_alltypes ENGINE = Memory AS
SELECT * REPLACE(CAST(timestamp_col AS Nullable(DateTime)) AS timestamp_col)
FROM file('ibis/functional_alltypes.parquet', 'Parquet');

CREATE OR REPLACE TABLE tzone (
    ts Nullable(DateTime),
    key Nullable(String),
    value Nullable(Float64)
) ENGINE = Memory;

CREATE OR REPLACE TABLE array_types (
    x Array(Nullable(Int64)),
    y Array(Nullable(String)),
    z Array(Nullable(Float64)),
    grouper Nullable(String),
    scalar_column Nullable(Float64),
    multi_dim Array(Array(Nullable(Int64)))
) ENGINE = Memory;

INSERT INTO array_types VALUES
    ([1, 2, 3], ['a', 'b', 'c'], [1.0, 2.0, 3.0], 'a', 1.0, [[], [1, 2, 3], []]),
    ([4, 5], ['d', 'e'], [4.0, 5.0], 'a', 2.0, []),
    ([6, NULL], ['f', NULL], [6.0, NULL], 'a', 3.0, [[], [], []]),
    ([NULL, 1, NULL], [NULL, 'a', NULL], [], 'b', 4.0, [[1], [2], [], [3, 4, 5]]),
    ([2, NULL, 3], ['b', NULL, 'c'], NULL, 'b', 5.0, []),
    ([4, NULL, NULL, 5], ['d', NULL, NULL, 'e'], [4.0, NULL, NULL, 5.0], 'c', 6.0, [[1, 2, 3]]);

CREATE OR REPLACE TABLE time_df1 (
    time Int64,
    value Nullable(Float64),
    key Nullable(String)
) ENGINE = Memory;
INSERT INTO time_df1 VALUES
    (1, 1.0, 'x'),
    (20, 20.0, 'x'),
    (30, 30.0, 'x'),
    (40, 40.0, 'x'),
    (50, 50.0, 'x');

CREATE OR REPLACE TABLE time_df2 (
    time Int64,
    value Nullable(Float64),
    key Nullable(String)
) ENGINE = Memory;
INSERT INTO time_df2 VALUES
    (19, 19.0, 'x'),
    (21, 21.0, 'x'),
    (39, 39.0, 'x'),
    (49, 49.0, 'x'),
    (1000, 1000.0, 'x');

CREATE OR REPLACE TABLE struct (
    abc Tuple(
        a Nullable(Float64),
        b Nullable(String),
        c Nullable(Int64)
    )
) ENGINE = Memory;

-- NULL is the same as tuple(NULL, NULL, NULL) because clickhouse doesn't
-- support Nullable(Tuple(...))
INSERT INTO struct VALUES
    (tuple(1.0, 'banana', 2)),
    (tuple(2.0, 'apple', 3)),
    (tuple(3.0, 'orange', 4)),
    (tuple(NULL, 'banana', 2)),
    (tuple(2.0, NULL, 3)),
    (tuple(NULL, NULL, NULL)),
    (tuple(3.0, 'orange', NULL));

CREATE OR REPLACE TABLE map (kv Map(String, Nullable(Int64))) ENGINE = Memory;

INSERT INTO map VALUES
    (map('a', 1, 'b', 2, 'c', 3)),
    (map('d', 4, 'e', 5, 'c', 6));

CREATE OR REPLACE TABLE win (g String, x Int64, y Int64) ENGINE = Memory;
INSERT INTO win VALUES
    ('a', 0, 3),
    ('a', 1, 2),
    ('a', 2, 0),
    ('a', 3, 1),
    ('a', 4, 1);
