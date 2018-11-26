.. _documentation:

===========================
Documentation
===========================

Documentation is provided as restructured text and as docstrings within the code. HTML pages are formatted using
Sphinx_.

.. _Sphinx: http://www.sphinx-doc.org

We follow standard recommendations for `python documentation`_ and the `numpy conventions`_.

.. _python documentation: https://www.python.org/dev/peps/pep-0008/
.. _numpy conventions: https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt

Update documentation
--------------------

To update the documentation from sources delete ``/docs/sources/generated`` and run::

    sphinx-html -b html -E YOUR_PATH\Surepy\docs YOUR_PATH\Surepy\docs\_build

