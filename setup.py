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

from setuptools import setup

setup(name='pyhfo_detect',
      version='0.1',
      install_requires=['pandas','pyedflib'],
      description='Collection of (semi)automated detectors of HFO',
      #url='http://github.com/cimbi/pancircs',
      #author='Jan Cimbalnik and collaborators',
      #author_email='jan.cimbalnik@fnusa.cz, jan.cimbalnik@mayo.edu',
      #license='BSD 3.0',
      packages=['pyhfo_detect'],
      keywords='hfo automated detection',
#      classifiers=[
#          'Development Status :: 2 - Pre-Alpha',
#          'Intended Audience :: Science/Research',
#          'License :: OSI Approved :: BSD License',
#          'Programming Language :: Python :: 3.4',
#          'Topic :: Scientific/Engineering :: Visualization'],
      zip_safe=False)
