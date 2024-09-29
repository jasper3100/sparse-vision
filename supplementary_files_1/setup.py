from setuptools import setup, find_packages

# Code to setup subdirectories as modules
# In the subdirectories as well as the root directory,
# there should be __ini__.py files (which can be empty)

setup(
    name='setup',
    version='0.1',
    packages=find_packages(),
    # Alternatively, specify individual subdirectories
    # packages=['subdirectory1', 'subdirectory2'],
)