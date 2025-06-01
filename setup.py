from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="ColorectalCancerProject",
    version="0.1",
    author="Jalaj",
    packages=find_packages(),
    install_requires=requirements,
)