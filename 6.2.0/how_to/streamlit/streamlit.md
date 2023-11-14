# Write a Streamlit app with Ibis

Streamlit + Ibis = :heart:

Ibis supports the [streamlit `experimental_connection` interface](https://blog.streamlit.io/introducing-st-experimental_connection/), making it easier than ever to combine the powers of both tools!

Check out the example application below that shows the top N ingredients from a corpus of recipes using [the ClickHouse backend](../backends/clickhouse.md)!

<div class="streamlit-app">
  <iframe class="streamlit-app-inner" src="https://ibis-example.streamlit.app/?embedded=true"></iframe>
</div>

And here's the source code for the application:

??? example "Source code"

    ```python title="example_streamlit_app.py"
    --8<-- "docs/example_streamlit_app/example_streamlit_app.py"
    ```
