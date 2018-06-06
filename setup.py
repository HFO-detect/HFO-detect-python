# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 09:59:27 2016

Setup file for the biosigpy package.

Ing.,Mgr. (MSc.) Jan Cimbálník
Biomedical engineering
International Clinical Research Center
St. Anne's University Hospital in Brno
Czech Republic
&
Mayo systems electrophysiology lab
Mayo Clinic
200 1st St SW
Rochester, MN
United States
"""

import os
from setuptools import setup

def package_tree(pkgroot):
    path = os.path.dirname(__file__)
    subdirs = [os.path.relpath(i[0], path).replace(os.path.sep, '.')
               for i in os.walk(os.path.join(path, pkgroot))
               if '__init__.py' in i[2]]
    return subdirs

setup(name='pyhfo_detect',
      version='0.2.1',
      install_requires=['pandas','numpy','scipy'],
      description='Collection of (semi)automated detectors of HFO',
      url='http://github.com/cimbi/HFO-detect/HFO-detect-python',
      author='Jan Cimbalnik and collaborators',
      author_email='jan.cimbalnik@fnusa.cz, jan.cimbalnik@mayo.edu',
      license='BSD 3.0',
      #packages=find_packages('pyhfo_detect'),
      packages=package_tree('pyhfo_detect'),
      package_data={'pyhfo_detect':['core/*.pkl']},
      keywords='hfo automated detection',
#      classifiers=[
#          'Development Status :: 2 - Pre-Alpha',
#          'Intended Audience :: Science/Research',
#          'License :: OSI Approved :: BSD License',
#          'Programming Language :: Python :: 3.4',
#          'Topic :: Scientific/Engineering :: Visualization'],
      zip_safe=False)
