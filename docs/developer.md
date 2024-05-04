
# How to upload:

* commit all changes
* change/propagate version number in main `__init__.py`, flit takes it from there
* build your package (remove previous builds in `/dist` folder)  
  `python3 -m build`
* upload with twine to pypi (API key located in `.pypirc`)  
  `python3 -m twine upload --repository pypi dist/*`
