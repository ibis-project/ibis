# How to Use `register` to load external data files with the DuckDB backend

<!-- prettier-ignore-start -->
Here we use the `register` method to load external data files and join them.
<!-- prettier-ignore-end -->

We're going to download one month of [NYC Taxi
data](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page) in
`parquet` format and also download the "Taxi Zone Lookup Table" which is a `csv`

https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2022-01.parquet
https://d37ci6vzurychx.cloudfront.net/misc/taxi+_zone_lookup.csv

Create an in-memory DuckDB connection via `ibis`

```python
>>> import ibis
>>> con = ibis.duckdb.connect()  # in-memory database
>>> con.list_tables()
[]
```

Now we call `register` with the filepath (the `table_name` argument is optional,
if it isn't specified, Ibis will use the filename minus the extension)

```python
>>> con.register("taxi+_zone_lookup.csv", table_name="taxi_zone_lookup")
AlchemyTable: taxi+_zone_lookup
  LocationID   int32
  Borough      string
  Zone         string
  service_zone string

>>> con.register("green_tripdata_2022-01.parquet", table_name="tripdata")
AlchemyTable: green_tripdata_2022_01
  VendorID              int64
  lpep_pickup_datetime  timestamp
  lpep_dropoff_datetime timestamp
  store_and_fwd_flag    string
  RatecodeID            float64
  PULocationID          int64
  DOLocationID          int64
  passenger_count       float64
  trip_distance         float64
  fare_amount           float64
  extra                 float64
  mta_tax               float64
  tip_amount            float64
  tolls_amount          float64
  ehail_fee             int32
  improvement_surcharge float64
  total_amount          float64
  payment_type          float64
  trip_type             float64
  congestion_surcharge  float64
>>> con.list_tables()
['tripdata, 'taxi_zone_lookup']
```

We now have a schema parsed from the files and corresponding tables (they are
actually `views` that are lazily-loaded) are available.

Now we can interact with these tables just like a table or view in any backend
connection:

```python
>>> lookup = con.table("taxi_zone_lookup")
>>> tripdata = con.table("tripdata")

>>> tripdata.columns
['VendorID', 'lpep_pickup_datetime', 'lpep_dropoff_datetime', 'store_and_fwd_flag', 'RatecodeID', 'PULocationID', 'DOLocationID', 'passenger_count', 'trip_distance', 'fare_amount', 'extra', 'mta_tax', 'tip_amount', 'tolls_amount', 'ehail_fee', 'improvement_surcharge', 'total_amount', 'payment_type', 'trip_type', 'congestion_surcharge']

>>> lookup.columns
['LocationID', 'Borough', 'Zone', 'service_zone']
```

We can grab a small subset of the `tripdata` columns and then join them to the
`lookup` table to get human-readable values for the pickup locations:

```python
>>> ibis.options.interactive = True

>>> tripdata = tripdata[["lpep_pickup_datetime", "PULocationID"]]

>>> tripdata.head()
  lpep_pickup_datetime  PULocationID
0  2022-01-01 00:14:21            42
1  2022-01-01 00:20:55           116
2  2022-01-01 00:57:02            41
3  2022-01-01 00:07:42           181
4  2022-01-01 00:07:50            33

>>> tripdata.join(lookup, tripdata.PULocationID == lookup.LocationID).head()
  lpep_pickup_datetime  PULocationID  LocationID    Borough                  Zone service_zone
0  2022-01-01 00:14:21            42          42  Manhattan  Central Harlem North    Boro Zone
1  2022-01-01 00:20:55           116         116  Manhattan      Hamilton Heights    Boro Zone
2  2022-01-01 00:57:02            41          41  Manhattan        Central Harlem    Boro Zone
3  2022-01-01 00:07:42           181         181   Brooklyn            Park Slope    Boro Zone
4  2022-01-01 00:07:50            33          33   Brooklyn      Brooklyn Heights    Boro Zone
```

That's it!

Ibis+duckdb currently supports registering `parquet`, `csv`, and `csv.gz`.

You can pass in the filename and the filetype will be inferred from the extension, or you can pass it explicitly using a file URI, e.g.

```python
con.register("csv://some_csv_file_without_an_extension")
con.register("csv.gz://a_compressed_csv_file.csv")
con.register("parquet://a_parquet_file_with_truncated_extension.parq")
```
