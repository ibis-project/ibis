# [{{ backend_name }}]({{ backend_url }})

{% if is_experimental %}
!!! experimental "{% if version_added %}New in v{{ version_added }}{% else %}Experimental{% endif %}"

    The {{ backend_name }} backend is experimental and is subject to backwards incompatible changes.

{% endif %}

{% if intro %}{{ intro }}{% endif %}

{% if not development_only %}

## Install

Install ibis and dependencies for the {{ backend_name }} backend:

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

    [Set up a development environment](../community/contribute/01_environment.md) to use this backend.

{% endif %}

## Connect

### API

Create a client by passing in {{ backend_param_style }} to `ibis.{{ backend_module }}.connect`.

<!-- prettier-ignore-start -->
See [`ibis.backends.{{ backend_module }}.Backend.do_connect`][ibis.backends.{{ backend_module }}.Backend.do_connect]
for connection parameter information.
<!-- prettier-ignore-end -->

<!-- prettier-ignore-start -->
!!! info "`ibis.{{ backend_module }}.connect` is a thin wrapper around [`ibis.backends.{{ backend_module }}.Backend.do_connect`][ibis.backends.{{ backend_module }}.Backend.do_connect]."
<!-- prettier-ignore-end -->

### Connection Parameters

<!-- prettier-ignore-start -->
::: ibis.backends.{{ backend_module }}.Backend.do_connect
    options:
      heading_level: 4
<!-- prettier-ignore-end -->

{% if not exclude_backend_api %}

## Backend API

<!-- prettier-ignore-start -->
::: ibis.backends.{{ backend_module }}.Backend
    options:
      heading_level: 3
      inherited_members: true
<!-- prettier-ignore-end -->

{% endif %}
