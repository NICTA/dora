""" Setup utility for the dora package. """
from setuptools import setup, find_packages
# from setuptools.command.test import test as TestCommand


setup(
    name='dora',
    version='0.1',
    description='Active sampling using a non-parametric regression model.',
    author="Alistair Reid and Simon O'Callaghan",
    author_email='alistair.reid@nicta.com.au',
    url='http://github.com/nicta/dora',
    packages=find_packages(),
    # cmdclass={
    #     'test': PyTest
    # },
    tests_require=['pytest'],
    install_requires=[
        'scipy >= 0.14.0',
        'numpy >= 1.8.2'
        # 'six >= 1.9.0',
        # NLopt >= 2.4.2
    ],
    dependency_links=[
        'git+https://github.com/nicta/revrand.git@master#egg=revrand'],
    extras_require={
        # 'nonlinear': ['NLopt'],
        'demos': [
            'unipath',
            'requests',
            # 'bdkd-external'
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
