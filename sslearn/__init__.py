# Open README.md and added to __doc__ for 
import os
if os.path.exists("../README.md"):
    with open("../README.md", "r") as f:
        __doc__ = f.read()
elif os.path.exists("README.md"):
    with open("README.md", "r") as f:
        __doc__ = f.read()
else:
    __doc__ = "Semi-Supervised Learning (SSL) is a Python package that provides tools to train and evaluate semi-supervised learning models."


__version__='1.0.5'
__AUTHOR__="José Luis Garrido-Labrador"  # Author of the package
__AUTHOR_EMAIL__="jlgarrido@ubu.es"  # Author's email
__URL__="https://pypi.org/project/sslearn/"


