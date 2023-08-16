DROP TABLE IF EXISTS functional_alltypes;

CREATE TABLE functional_alltypes (
    id INT,
    bool_col BOOLEAN,
    tinyint_col TINYINT,
    smallint_col SMALLINT,
    int_col INT,
    bigint_col BIGINT,
    float_col FLOAT,
    double_col DOUBLE,
    date_string_col VARCHAR,
    string_col VARCHAR,
    timestamp_col TIMESTAMP,
    year INT,
    month INT
) WITH (
    'connector' = 'filesystem',
    'path' = 'file:///{data_dir}/csv/functional_alltypes.csv',
    'format' = 'csv',
    'csv.ignore-parse-errors' = 'true'
);

DROP TABLE IF EXISTS diamonds;

CREATE TABLE diamonds (
    carat DOUBLE,
    cut VARCHAR,
    color VARCHAR,
    clarity VARCHAR,
    depth DOUBLE,
    table DOUBLE,
    price BIGINT,
    x DOUBLE,
    y DOUBLE,
    z DOUBLE
) WITH (
    'connector' = 'filesystem',
    'path' = 'file:///{data_dir}/csv/diamonds.csv',
    'format' = 'csv',
    'csv.ignore-parse-errors' = 'true'
);

DROP TABLE IF EXISTS batting;

CREATE TABLE batting (
    playerID VARCHAR,
    yearID BIGINT,
    stint BIGINT,
    teamID VARCHAR,
    lgID VARCHAR,
    G BIGINT,
    AB BIGINT,
    R BIGINT,
    H BIGINT,
    X2B BIGINT,
    X3B BIGINT,
    HR BIGINT,
    RBI BIGINT,
    SB BIGINT,
    CS BIGINT,
    BB BIGINT,
    SO BIGINT,
    IBB BIGINT,
    HBP BIGINT,
    SH BIGINT,
    SF BIGINT,
    GIDP BIGINT
) WITH (
    'connector' = 'filesystem',
    'path' = 'file:///{data_dir}/csv/batting.csv',
    'format' = 'csv',
    'csv.ignore-parse-errors' = 'true'
);

DROP TABLE IF EXISTS awards_players;

CREATE TABLE awards_players (
    playerID VARCHAR,
    awardID VARCHAR,
    yearID BIGINT,
    lgID VARCHAR,
    tie VARCHAR,
    notes VARCHAR
) WITH (
    'connector' = 'filesystem',
    'path' = 'file:///{data_dir}/csv/awards_players.csv',
    'format' = 'csv',
    'csv.ignore-parse-errors' = 'true'
);
