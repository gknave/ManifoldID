from setuptools import setup
setup(name='manifoldid',
    version='0.1',
	description='Package for finding attracting manifolds in phase space',
	url='',
	author='Gary Nave'
	author_email='gknave@vt.edu',
	packages=['manifoldid'],
	install_requires=[
	  'numpy',
	  'scipy',
	  'mayavi',
	]
	zip_safe=False)