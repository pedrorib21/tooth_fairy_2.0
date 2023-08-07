from setuptools import setup, find_packages

# Project metadata
name = "Remove Tooth"
version = "0.1.0"
description = "Training a PointNet that learns how to identify teeth. After identifying them you can remove them with our processing/meshes module.\
                Check how everythinh work in remove_tooth.ipynb"
author = "Pedro Ribeiro"
author_email = "pedrorib12345@gmail.com"
license = "MIT"

# Project dependencies from requirements.txt
with open("requirements.txt", "r", encoding="utf-8") as requirements_file:
    install_requires = [
        line.strip() for line in requirements_file if not line.startswith("#")
    ]

# Long description from README.md
with open("README.md", "r", encoding="utf-8") as readme_file:
    long_description = readme_file.read()

# Setup configuration
setup(
    name=name,
    version=version,
    description=description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=author,
    author_email=author_email,
    license=license,
    packages=find_packages(),
    install_requires=install_requires,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="your, keywords, here",
)
