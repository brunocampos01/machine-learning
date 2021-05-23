# Create packages to prepare deploy.
from io import open
from os import path

from setuptools import find_packages, setup


here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, '../README.md')) as f:
    long_description = f.read()

with open(path.join(here, '../requirements.txt')) as f:
    requirements = f.read()

setup(
    name=here,
    version='0.1',
    author='brunocampos01',
    author_email='brunocampos01@gmail.com',
    url="https://github.com/brunocampos01",
    description='PREPARE DEPLOY',
    long_description=long_description,
    license='MIT',
    packages=find_packages(),
    scripts=['environment/make',
             'environment/prepare_environment.py',
             'environment/show_config_environment.sh'
             'environment/show_struture_project.sh'],
    install_requires=requirements,
    classifiers=[
        'Development Status :: Production/Stable',
        'Environment :: Jupyter-notebook',
        'Environment :: Web Environment',
        'Intended Audience :: Developers',
        'Intended Audience :: Data Scientist',
        'License :: OSI Approved :: MIT',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: System :: Data Science',
    ],
)
