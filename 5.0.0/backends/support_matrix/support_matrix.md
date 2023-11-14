---
hide:
  - toc
---

# Operation Support Matrix

Backends are shown in descending order of the number of supported operations.

!!! tip "Backends with low coverage are good places to start contributing!"

    Each backend implements operations differently, but this is usually very
    similar to other backends. If you want to start contributing to ibis, it's
    a good idea to start by adding missing operations to backends that have low
    operation coverage.

<div class="streamlit-app">
  <iframe id="streamlit-app" src="https://ibis-project.streamlit.app/?embedded=true"></iframe>
</div>

!!! note "This app is built using [`streamlit`](https://streamlit.io/)"

    You can develop the app locally by editing `docs/backends/app/backend_info_app.py` and
    opening a PR with your changes.

    Test your changes locally by running

    ```sh
    $ streamlit run docs/backends/app/backend_info_app.py
    ```

    The changes will show up in the dev docs when your PR is merged!

## Raw Data

You can also download data from the above tables in [CSV format](./raw_support_matrix.csv).
