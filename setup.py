#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='pantheonrl',
      version='0.0.1',
      description='PantheonRL',
      author='',
      author_email='',
      packages=find_packages(),
      include_package_data=True,
      install_requires=[
        'flask==2.2.2; python_version < "3.13"',
        'flask>=3.1,<4; python_version >= "3.13"',
        'tensorflow==2.11.0; python_version < "3.11"',
        'tensorflow>=2.12,<2.20; python_version >= "3.11" and python_version < "3.13"',
        'tensorflow>=2.20,<2.22; python_version >= "3.13"',
        'torch==1.13.1; python_version < "3.11"',
        'torch>=2.1,<3; python_version >= "3.11" and python_version < "3.13"',
        'torch>=2.6,<3; python_version >= "3.13"',
        'tensorboard==2.11.0; python_version < "3.11"',
        'tensorboard>=2.12,<2.20; python_version >= "3.11" and python_version < "3.13"',
        'tensorboard>=2.20,<2.21; python_version >= "3.13"',
        'stable-baselines3==1.7.0; python_version < "3.11"',
        'stable-baselines3>=2.6,<3; python_version >= "3.13"',
        'stable-baselines3>=2.0,<3; python_version >= "3.11" and python_version < "3.13"',
        'gym==0.26.2; python_version >= "3.11"',
        'shimmy>=2,<3; python_version >= "3.13"',
        'scipy==1.7.3; python_version < "3.11"',
        'scipy>=1.11,<1.18; python_version >= "3.11" and python_version < "3.13"',
        'scipy>=1.15,<1.19; python_version >= "3.13"',
        'tqdm>=4.64.1,<5'
      ],
      )
