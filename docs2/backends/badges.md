{% if imports %} ![filebadge](https://img.shields.io/badge/Reads-{{ "%20|%20".join(sorted(imports)) }}-blue?style=flat-square) {% endif %}

{% if exports %} ![exportbadge](https://img.shields.io/badge/Exports-{{ "%20|%20".join(sorted(exports)) }}-orange?style=flat-square) {% endif %}
