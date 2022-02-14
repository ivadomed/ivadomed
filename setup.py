from codecs import open
from os import path
from setuptools import setup, find_packages
from platform import python_version, uname, mac_ver

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
    requirements = f.readlines()


def get_whl_info():
    """Returns the whl info to handpick torch whls.

    whl format: {dist}-{version}(-{build})?-{python_tag}-{abi_tag}-{platform}.whl

    Returns:
        str, str, str: python_tag, abi_tag, platform.
    """
    py_version = python_version()[0:3].replace('.', '')
    python_tag = f'cp{py_version}'

    if py_version == '36' or py_version == '37':
        abi_tag = f'{python_tag}m'
    elif py_version == '38' or py_version == '39':
        abi_tag = python_tag

    sysinfo = uname()
    if sysinfo.system == 'Linux':
        if sysinfo.machine == 'x86_64':
            platform = 'linux_x86_64'
            return python_tag, abi_tag, platform
        elif sysinfo.machine == 'aarch64':
            raise ValueError(f"ivadomed doesn't handpick torch whls for {sysinfo.machine}, \
            install torch==1.8.0+cpu, torchvision==0.9.0+cpu or \
            torch==1.8.1+cu111, torchvision==0.9.1+cu111 from \
            https://pytorch.org/get-started/previous-versions/")
    elif sysinfo.system == 'Windows' and sysinfo.machine == 'AMD64':
        platform = 'win_amd64'
        return python_tag, abi_tag, platform
    elif sysinfo.system == 'Darwin':
        # TODO: update mac conditions as we upgrade to higher torch versions
        # with separate binaries for 10 and 11 versions   
        # macinfo = mac_ver()
        # if macinfo[0].startswith('10') and macinfo[2] == 'x86_64':
        platform = 'macosx_10_9_x86_64'
        return python_tag, abi_tag, platform 


def get_torch_whls(backend):
    """Handpicks whls for torch and torchvision based on the backend.

    Args:
        backend (str): cpu or gpu

    Returns:
        list[str]: List of handpicked torch et al. whls
    """
    python_tag, abi_tag, platform = get_whl_info()
    if backend == 'cpu':
        if platform.startswith('macosx'):
            _torch_whl = f'-{python_tag}-none-{platform}.whl'
            torch_whl = f'torch@https://download.pytorch.org/whl/cpu/torch-1.8.0{_torch_whl}'
            _torchvision_whl = f'-{python_tag}-{abi_tag}-{platform}.whl'
            torchvision_whl = f'torchvision@https://download.pytorch.org/whl/torchvision-0.9.0{_torchvision_whl}'
            return torch_whl, torchvision_whl
        else:
            whl = f'%2B{backend}-{python_tag}-{abi_tag}-{platform}.whl'
            torch_whl = f'torch@https://download.pytorch.org/whl/cpu/torch-1.8.0{whl}'
            torchvision_whl = f'torchvision@https://download.pytorch.org/whl/cpu/torchvision-0.9.0{whl}'
            return torch_whl, torchvision_whl
    elif backend == 'gpu':
        if platform.startswith('macosx'):
            print("MacOS Binaries don't support CUDA, install from source if CUDA is needed")
        else:
            backend = 'cu111'
            whl = f'%2B{backend}-{python_tag}-{abi_tag}-{platform}.whl'
            torch_whl = f'torch@https://download.pytorch.org/whl/cu111/torch-1.8.1{whl}'
            torchvision_whl = f'torchvision@https://download.pytorch.org/whl/cu111/torchvision-0.9.1{whl}'
            return torch_whl, torchvision_whl


extra_requirements = {
    'cpu': [
        requirements,
        get_torch_whls(backend='cpu')
    ],
    'gpu': [
        requirements,
        get_torch_whls(backend='gpu'),
    ],
    'docs': [
        # pin sphinx to match what RTD uses:
        # https://github.com/readthedocs/readthedocs.org/blob/ecac31de54bbb2c100f933e86eb22b0f4389ba84/requirements/pip.txt#L16
        'sphinx',
        'sphinx_rtd_theme',
        'sphinx-tabs==3.2.0',
        'sphinx-toolbox==2.15.2',
        'sphinx-jsonschema~=1.16',
    ],
    'tests': [
        'pytest~=6.2',
        'pytest-cov',
        'pytest-ordering~=0.6',
        'pytest-console-scripts~=1.1',
        'coverage',
        'coveralls',
    ]
}

extra_requirements['dev'] = [
    extra_requirements['cpu'],
    extra_requirements['docs'],
    extra_requirements['tests'],
    'pypandoc',
    'pre-commit>=2.10.1',
    'flake8'
    ]

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
    extras_require=extra_requirements,
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
