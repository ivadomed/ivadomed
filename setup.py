from codecs import open
# TODO: use pathlib
from os import path
from setuptools import setup, find_packages
from distutils.core import Command
import subprocess
from sys import platform
import shlex

here = path.abspath(path.dirname(__file__))

# Get the long description from README.md file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Get release version
with open(path.join(here, 'ivadomed', 'version.txt')) as f:
    version = f.read().strip()

# Manually specified, more generic version of the software.
# See: https://stackoverflow.com/a/49684835
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line]

class InstallTorch(Command):
    description = "Installs the PyTorch utilities"
    user_options = [
        ('backend=', '0', '1'),
    ]
    
    def initialize_options(self):
        self.backend = '0'
    
    def finalize_options(self):
        assert self.backend in ('0', '1'), 'Invalid backend!'
        
    def run(self):
        if self.backend == '0' and platform == 'darwin':
            command = f'pip install torch==1.8.0 torchvision==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html'
        elif self.backend == '0' and platform != 'darwin':
            command = f'pip install torch==1.8.0+cpu torchvision==0.9.0+cpu -f https://download.pytorch.org/whl/torch_stable.html'
        elif self.backend == '1':
            command = f'pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html'
        
        # TODO: inside try except
        subprocess.run(shlex.split(command))

extra_requirements = {}

extra_requirements['dev'] = sorted(
        'pytest~=6.2',
        'pytest-cov',
        'pytest-ordering~=0.6',
        'sphinx',
        'flake8',
        'coverage',
        'coveralls',
        'pypandoc',
        'sphinx_rtd_theme',
        'sphinx-jsonschema~=1.16',
        'pytest-console-scripts~=1.1',
        'pre-commit>=2.10.1',
        'sphinx-tabs==3.2.0'
)
# {
#         'docs': [  # pin sphinx to match what RTD uses:
#             # https://github.com/readthedocs/readthedocs.org/blob/ecac31de54bbb2c100f933e86eb22b0f4389ba84/requirements/pip.txt#L16
#             'sphinx==4.2.0',
#             'sphinx-rtd-theme<0.5',


setup(
    name='ivadomed',
    version=version,
    description='Feature conditioning for IVADO medical imaging project.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/neuropoly/ivadomed',
    author='NeuroPoly and Mila',
    author_email='none@none.com',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
    ],
    python_requires='>=3.6,<3.10',
    packages=find_packages(exclude=['docs', 'tests']),
    include_package_data=True,
    install_requires=requirements,
    extras_require=extra_requirements,
    cmdclass={
        'install': InstallTorch,
    },
    entry_points={
        'console_scripts': [
            'ivadomed=ivadomed.main:run_main',
            'ivadomed_prepare_dataset_vertebral_labeling=ivadomed.scripts.prepare_dataset_vertebral_labeling:main',
            'ivadomed_automate_training=ivadomed.scripts.automate_training:main',
            'ivadomed_compare_models=ivadomed.scripts.compare_models:main',
            'ivadomed_visualize_transforms=ivadomed.scripts.visualize_transforms:main',
            'ivadomed_convert_to_onnx=ivadomed.scripts.convert_to_onnx:main',
            'ivadomed_extract_small_dataset=ivadomed.scripts.extract_small_dataset:main',
            'ivadomed_download_data=ivadomed.scripts.download_data:main',
            'ivadomed_training_curve=ivadomed.scripts.training_curve:main',
            'ivadomed_visualize_and_compare_testing_models=ivadomed.scripts.visualize_and_compare_testing_models:main'
        ],
    },
)
