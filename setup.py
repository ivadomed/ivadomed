from setuptools import setup, find_packages
from codecs import open
from os import path

import ivadomed

here = path.abspath(path.dirname(__file__))

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

#Get README
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='ivadomed',
    version=ivadomed.__version__,
    description='Feature conditioning for IVADO medical imaging project.',
    long_description=long_description,
    long_description_content_type='text/markdown',
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
