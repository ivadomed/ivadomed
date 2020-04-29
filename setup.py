from setuptools import setup, find_packages
from codecs import open
from os import path

import ivadomed

here = path.abspath(path.dirname(__file__))

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='ivadomed',
    version=ivadomed.__version__,
    description='Feature conditioning for IVADO medical imaging project.',
    url='https://github.com/neuropoly/ivado-medical-imaging',
    author='NeuroPoly and Mila',
    author_email='none@none.com',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
    ],
    packages=find_packages(exclude=['docs', 'tests']),
    include_package_data=True,
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'ivadomed=ivadomed.main:run_main',
        ],
    },
)
