{{ fullname | escape | underline }}

.. currentmodule:: {{ fullname }}

.. automodule:: {{ fullname }}

{% if classes %}
.. rubric:: Classes

.. autosummary::
    :toctree: ./{{ fullname }}
    {% for class in classes %}
    {{ class }}
    {% endfor %}

{% endif %}


{% if functions %}
.. rubric:: Functions

.. autosummary::
    :toctree: ./{{ fullname }}
    {% for function in functions %}
    {{ function }}
    {% endfor %}

{% endif %}
