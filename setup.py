#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name='coconuts',
    version='0.0.1',
    description='Pytorch wrapper implementing Bananas API',
    long_description='Pytorch wrapper implementing Bananas API',
    author='owahltinez',
    author_email='oscar@wahltinez.org',
    url='https://github.com/owahltinez/coconuts',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'bananas @ https://api.github.com/repos/owahltinez/bananas/tarball/master',
        'torch'
    ],
    license='MIT',
    zip_safe=False,
    keywords=['ML'],
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7'
    ],
    test_suite='tests',
    tests_require=[]
)
