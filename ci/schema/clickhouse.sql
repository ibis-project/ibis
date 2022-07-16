CREATE TABLE diamonds (
    carat Nullable(Float64),
    cut Nullable(String),
    color Nullable(String),
    clarity Nullable(String),
    depth Nullable(Float64),
    `table` Nullable(Float64),
    price Nullable(Int64),
    x Nullable(Float64),
    y Nullable(Float64),
    z Nullable(Float64)
) ENGINE = Memory;

CREATE TABLE batting (
    `playerID` Nullable(String),
    `yearID` Nullable(Int64),
    stint Nullable(Int64),
    `teamID` Nullable(String),
    `lgID` Nullable(String),
    `G` Nullable(Int64),
    `AB` Nullable(Int64),
    `R` Nullable(Int64),
    `H` Nullable(Int64),
    `X2B` Nullable(Int64),
    `X3B` Nullable(Int64),
    `HR` Nullable(Int64),
    `RBI` Nullable(Int64),
    `SB` Nullable(Int64),
    `CS` Nullable(Int64),
    `BB` Nullable(Int64),
    `SO` Nullable(Int64),
    `IBB` Nullable(Int64),
    `HBP` Nullable(Int64),
    `SH` Nullable(Int64),
    `SF` Nullable(Int64),
    `GIDP` Nullable(Int64)
) ENGINE = Memory;

CREATE TABLE awards_players (
    `playerID` Nullable(String),
    `awardID` Nullable(String),
    `yearID` Nullable(Int64),
    `lgID` Nullable(String),
    tie Nullable(String),
    notes Nullable(String)
) ENGINE = Memory;

CREATE TABLE functional_alltypes (
    `index` Nullable(Int64),
    `Unnamed: 0` Nullable(Int64),
    id Nullable(Int32),
    bool_col Nullable(UInt8),
    tinyint_col Nullable(Int8),
    smallint_col Nullable(Int16),
    int_col Nullable(Int32),
    bigint_col Nullable(Int64),
    float_col Nullable(Float32),
    double_col Nullable(Float64),
    date_string_col Nullable(String),
    string_col Nullable(String),
    timestamp_col Nullable(DateTime),
    year Nullable(Int32),
    month Nullable(Int32)
) ENGINE = Memory;

CREATE TABLE tzone (
    ts Nullable(DateTime),
    key Nullable(String),
    value Nullable(Float64)
) ENGINE = Memory;

CREATE TABLE IF NOT EXISTS array_types (
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


CREATE TABLE IF NOT EXISTS struct (
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
