#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 20:13:21 2020

@author: r17935avinash
"""

from distutils.core import setup
setup(
  name = 'HindiNLP',         # How you named your package folder (MyLib)
  packages = ['HindiNLP'],   # Chose the same as "name"
  version = '0.1',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'Specialized NLP library which provides tools to perform basic NLP tasks on Hindi Datasets',   # Give a short description about your library
  author = 'avinash_swaminathan',                   # Type in your name
  author_email = 's.avinash.it.17@nsit.net.in',      # Type in your E-Mail
  url = 'https://github.com/avinsit123/HindiNLPTools',   # Provide either the link to your github or to your website
  download_url = "https://github.com/avinsit123/HindiNLPTools/archive/v1.0.tar.gz",    # I explain this later on
  keywords = ['Hindi-NLP', 'NER', 'Classifier'],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
          'flair',
          'torch',
           'typing',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package     # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license     #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.6',
  ],
)

