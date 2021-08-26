import os

from setuptools import find_packages, setup

with open(os.path.join(os.getcwd(), "requirements.txt")) as f:
    install_requires = f.read().splitlines()

with open(os.path.join(os.getcwd(), "requirements-dev.txt")) as f:
    install_requires_dev = f.read().splitlines()

with open(os.path.join(os.getcwd(), "requirements-examples.txt")) as f:
    install_requires_examples = f.read().splitlines()

with open("VERSION") as f:
    version = f.read().strip()

if os.getenv("PRERELEASE"):
    version += os.getenv("PRERELEASE")


setup(
    name="afterglow",
    version=version,
    packages=find_packages(),
    install_requires=install_requires,
    extras_require={"dev": install_requires_dev, "examples": install_requires_examples},
)
