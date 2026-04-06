Installation
============

From PyPI
---------

Install qitesse from PyPI:

.. code-block:: bash

   pip install qitesse

From Source
-----------

To build qitesse from this repository:

.. code-block:: bash

   git clone https://github.com/OsamaMIT/qitesse.git
   cd qitesse
   pip install maturin
   maturin develop --release

Building The Docs Locally
-------------------------

The API reference imports the installed qitesse extension and generates class pages automatically with Sphinx.

Build the extension first, then build the docs:

.. code-block:: bash

   maturin develop --release
   pip install -e .[docs]
   sphinx-build -b html docs docs/_build/html

Read The Docs uses the checked-in :code:`.readthedocs.yaml` file and the :code:`docs` dependency extra to do the same thing automatically.
