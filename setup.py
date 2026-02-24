# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 16:48:59 2023

@author: chenruipu
"""

import setuptools

setuptools.setup(
    name="celllingo",
    version="1.0",
    author="Chen Ruipu",
    author_email="chencuge@live.com",
    description="cross species cell-tyep annotation with gene ontology",
    url= "https://github.com/chenruipu/CellLingo-main",
    # long_description="CellLingo",
    # long_description_content_type="CellLingo",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['scanpy',
                      'dgl >=1',
                      'torch >=1.13,<2']
)



