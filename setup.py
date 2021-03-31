from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='braingeneerspy',
    version='0.0.7',
    description='Braingeneers Python utilities',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/braingeneers/braingeneerspy',
    author='Braingeneers',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Braingeneers',
        'Programming Language :: Python :: 3',
        'License :: MIT'
    ],
    packages=find_packages(exclude=()),
    install_requires=['numpy', 'scipy', 'requests'],
    include_package_data=True)
