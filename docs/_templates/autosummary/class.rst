{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :undoc-members:
   :show-inheritance:

{% if attributes %}
Attributes
----------

{% for item in attributes %}
- ``{{ item }}``
{%- endfor %}
{% endif %}

{% if methods %}
Methods
-------

{% for item in methods %}
{% if item != '__init__' %}
- ``{{ item }}``
{% endif %}
{%- endfor %}
{% endif %}
