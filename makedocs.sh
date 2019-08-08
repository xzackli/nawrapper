#python setup.py install
sphinx-apidoc -f -o docs/source nawrapper
cd docs
make html
#make latex
#make latexpdf
#cp build/latex/nawrapper.pdf .
