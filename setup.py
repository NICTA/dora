""" Setup utility for the dora package. """
from setuptools import setup, find_packages
# from setuptools.command.test import test as TestCommand


setup(
    name='dora',
    version='0.1',
    description='Active sampling using a non-parametric regression model.',
    url='http://github.com/nicta/dora',
    packages=find_packages(),
    # cmdclass={
    #     'test': PyTest
    # },
    tests_require=['pytest'],
    install_requires=[
        'scipy >= 0.14.0',
        'numpy >= 1.8.2',
        'revrand == 0.6.5',
        'jupyter >= 1.0.0',
        'matplotlib >= 2.0.2',
        'flask',
        'visvis',
        'requests'
    ],
    dependency_links=[
        'git+https://github.com/nicta/revrand.git@v0.6.5#egg=revrand-0.6.5'],
    extras_require={
        'demos': [
            'unipath',
            'requests',
        ],
    },
    license="Apache Software License 2.0",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Operating System :: POSIX",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.4",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis"
    ]
)
