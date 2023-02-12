from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='rfm',
    version='1.0',
    author='Adityanarayanan Radhakrishnan, Daniel Beaglehole, Parthe Pandit',
    author_email='aradha@mit.edu, dbeaglehole@ucsd.edu, parthepandit@ucsd.edu',
    description='Recursive procedure for Task-specific Feature Learning with Kernel predictors',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/aradha/recursive_feature_machines',
    project_urls = {
        "Bug Tracker": "https://github.com/aradha/recursive_feature_machines/issues"
    },
    license='MIT license',
    packages=find_packages(),
    install_requires=[
      'torchvision>=0.14',
      'hickle>=5.0',
      'tqdm'
    ],
)
