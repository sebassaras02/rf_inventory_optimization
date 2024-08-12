from setuptools import setup, find_packages

def parse_requirements(filename):
    with open(filename, "r") as file:
        return file.read().splitlines()
    
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="RLInventoryOpt",
    version="0.1.0",
    description="A Python library for inventory optimization with reinforcement learning using Q-Learning",
    long_description=long_description,
    author="Sebastian Sarasti",
    author_email="sebitas.alejo@hotmail.com",
    url="https://github.com/sebassaras02/rf_inventory_optimization",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=parse_requirements("requirements.txt"), 
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
