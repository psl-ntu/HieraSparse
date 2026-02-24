from pathlib import Path

from setuptools import find_packages, setup


def read_requirements(file):
    path = Path(__file__).parent / file
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]


install_requires = read_requirements("requirements.txt")
long_description = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

setup(
    name="hierasparse",
    version="0.1.0",
    description="HieraSparse: Hierarchical Semi-Structured KV-Cache Attention on Sparse Tensor Core",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="psl-hpc",
    url="https://github.com/psl-hpc/hierasparse",
    packages=find_packages(),
    include_package_data=True,
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries",
    ],
    python_requires=">=3.10",
    keywords="machine-learning attention sparse tensor",
)
