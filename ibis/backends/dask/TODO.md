# TODO

Double check conversions in `constants.py`

# Things I know are broken

## Broken in excution

- Windowing
- Interpolation for quantile
- Better join performance
- fix implementation of execute_materialized_join
- execute_substring_series_series
- execute_node_struct_field_series_group_by
- Any aggregation involving `SeriesGroupBy` is fairly broken
  - execute_binary_op_series_group_by
  - execute_binary_op_series_gb_simple
- execute_aggregation_dataframe
- execute_selection_dataframe and compute_sorted_frame are fairly broken

## Broken elsewhere

- create_table in client.py


# Also clean up

- Leftover bad calls directly to `dd.Series` and `dd.DataFrame`, litered through generic.py
