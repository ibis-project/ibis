# [{{ backend_name }}]({{ backend_url }})

{% if is_experimental %}
!!! experimental "{% if version_added %}New in v{{ version_added }}{% else %}Experimental{% endif %}"

    The {{ backend_name }} backend is experimental and is subject to backwards incompatible changes.

{% endif %}

{% if intro %}{{ intro }}{% endif %}

{% if not development_only %}

## Install

Install dependencies for the {{ backend_name }} backend:

=== "pip"

    ```sh
    pip install 'ibis-framework{% if not is_core %}[{{ backend_module }}]{% endif %}'
    ```

{% for mgr in ["conda", "mamba"] %}
=== "{{ mgr }}"

    ```sh
    {{ mgr }} install -c conda-forge ibis-{% if is_core %}framework{% else %}{{ backend_module }}{% endif %}
    ```

{% endfor %}

{% else %}
!!! info "The {{ backend_name }} backend isn't released yet!"

    [Set up a development environment](/contribute/01_environment) to use this backend.

{% endif %}

## Connect

<!-- prettier-ignore-start -->
Create a client by passing in {{ backend_param_style }} to [`ibis.{{ backend_module }}.connect`][ibis.backends.{{ backend_module }}.{{ do_connect_base or "Backend" }}.do_connect]:
<!-- prettier-ignore-end -->

```python
{{ backend_connection_example }}
```
