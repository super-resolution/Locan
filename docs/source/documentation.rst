.. _documentation:

===========================
Documentation
===========================

Documentation is provided as restructured text, myst markdown, and as docstrings within the code.
HTML pages and other documentation formats are build using Sphinx_.

.. _Sphinx: http://www.sphinx-doc.org

We follow standard recommendations for `python documentation`_ and the `numpy conventions`_.

.. _python documentation: https://www.python.org/dev/peps/pep-0008/
.. _numpy conventions: https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard

Update documentation
--------------------

To update the documentation from sources delete ``/docs/sources/generated`` and run::

    sphinx-build -b html -E YOUR_PATH\Locan\docs YOUR_PATH\Locan\docs\_build


Example docstring
-----------------
We try to follow standard docstrings as illustrated here::

    def function(par=None, **kwargs):
        """
        Short title.

        Long description about parameter `par` with some literature reference [1]_.

        This paragraph might describe some ``code`` and include an equation:

        .. math::

            \\f(x) = \\frac{a}{(b + c)}

        Parameters
        ----------
        par : None, other type, UserClass
            Description
        kwargs : dict
            Parameters passed to some other documented function :func:`function_name`

        Returns
        -------
        None

        Examples
        --------
        >>> 1 + 2
        3

        See Also
        --------
        :func:`locan.tests.test`

        Note
        ----
        Whatever there is to note.

        References
        ----------
        .. [1] <authors>, <title>. <journal> <year>, <volume>:<pages>. <doi>
        """
        return None



    class SomeClass:
        """
        Short title.

        Long description.

        Parameters
        ----------
        par : None, other type
            Description
        kwargs : dict
            Parameters passed to some other documented function :func:`function_name`

        Attributes
        ----------
        arg : None, other type
            Description
        """
        return None

