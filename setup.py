from setuptools import setup, find_packages

setup(
    name='easy_image_classifier',
    version='1.0',
    packages=find_packages(),
    install_requires=[
    'torch',
    'torchvision',
    'numpy',
    'matplotlib',
    'Pillow'
    ],
)