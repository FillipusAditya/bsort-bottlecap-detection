from setuptools import setup, find_packages

setup(
    name="bsort",
    version="0.1",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "ultralytics",
        "click",
        "pyyaml",
    ],
    entry_points={
        "console_scripts": [
            "bsort=bsort.cli:cli",
        ],
    },
)
