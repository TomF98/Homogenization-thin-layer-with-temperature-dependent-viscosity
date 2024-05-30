==========================================================================================
Implementation for the Paper: Analysis and Simulation of a Coupled Fluid-Heat System in a 
Thin, Rough Layer
==========================================================================================

This repository contains the source codes for all simulations presented in the `paper`_.

To run the code, the following packages are mandatory:

- `FEniCS`_
- `Gmsh`_
- `SciPy`_
- `meshio`_

An example environment is provided in the ``environment.yml``.

.. _`paper`: ...
.. _`FEniCS`: https://fenicsproject.org/
.. _`Gmsh`: https://pypi.org/project/gmsh/
.. _`SciPy`: https://scipy.org/install/
.. _`meshio`: https://pypi.org/project/meshio/


Usage:
======

Inside the folder `MeshCreation` all files that handle the creation of the domains, 
meshes and reference cells are collected. Once a mesh has been created, 
it has to be transformed with meshio to be usable with FEniCS, see ``meshio_convert.py``. 
Afterward, the mesh can be used to simulate the problem with the code provided in 
`2DImplementation` or `3DImplementation`.  