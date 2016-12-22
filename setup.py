from setuptools import setup
setup(name='manifoldid',
    version='0.1',
	description='Scratch package for methods to identify manifolds, particularly within simple glider models.',
	url='https://github.com/gknave/ManifoldID',
	author='Gary Nave'
	author_email='gknave@vt.edu',
	packages=['manifoldid'],
	install_requires=[
	  'numpy',
	  'scipy',
	  'mayavi',
	]
	zip_safe=False)