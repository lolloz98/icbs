from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='icbs-lolloz98',
    version='0.0.1',
    description='Simple image cut and rebuild package. It allows to decide how to cut an image in overlapping squares. After having cut the image you can rebuild it (or you can do some processing before rebuilding it)',
    long_description=long_description,
    author='Lorenzo Carpaneto',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
	url="https://github.com/lolloz98/icbs",
    python_requires='>=3.6'
)