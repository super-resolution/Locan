.. _documentation:

===========================
Documentation
===========================

Documentation is provided as restructured text and as docstrings within the code. HTML pages are formatted using
Sphinx_.

.. _Sphinx: http://www.sphinx-doc.org

We follow standard recommendations for `python documentation`_ and the `numpy conventions`_.

.. _python documentation: https://www.python.org/dev/peps/pep-0008/
.. _numpy conventions: https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard

Update documentation
--------------------

To update the documentation from sources delete ``/docs/sources/generated`` and run::

    sphinx-html -b html -E YOUR_PATH\Surepy\docs YOUR_PATH\Surepy\docs\_build


Example docstring
-----------------
We try to follow standard docstring as illustrated here::

    def function(par=None, **kwargs):
        """
        Short title.

        Long description.

        Parameters
        ----------
        par : None, other type
            Description

        Other parameters
        ----------------
        kwargs : dict
            Parameters passed to some other documented function

        Returns
        -------
        None

        Examples
        --------
        >>> 1 + 2
        3

        See Also
        --------
        :func:`surepy.tests.test`

        Notes
        -----
        Whatever there is to note.
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

        Other parameters
        ----------------
        kwargs : dict
            Parameters passed to some other documented function

        Attributes
        ----------
        arg : None, other type
            Description

        References
        ----------
        .. [1] some reference
        """
        return None

Examples
--------

In some examples we use the ipython directive sphinx extension:

.. ipython:: python

   x = 2
   print('hello')
   x*3
