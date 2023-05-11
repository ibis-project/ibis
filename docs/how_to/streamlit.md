# How to Write a Streamlit App with Ibis

Streamlit + Ibis = :heart:

Ibis supports the [streamlit `experimental_connection` interface](https://blog.streamlit.io/introducing-st-experimental_connection/), making it easier than ever to combine the powers of both tools!

Check out the example application below that shows the top N ingredients from a corpus of recipes using [the ClickHouse backend](../backends/ClickHouse.md)!

<div class="streamlit-app">
  <iframe class="streamlit-app-inner" src="https://cpcloud-docsexample-streamlit-appexample-streamlit-app-qau9wt.streamlit.app/?embedded=true"></iframe>
</div>

And here's the source code for the application:

??? example "Source code"

    ```python title="example_streamlit_app.py"
    --8<-- "docs/example_streamlit_app/example_streamlit_app.py"
    ```
