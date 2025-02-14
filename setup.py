from setuptools import setup, find_packages

setup(
    name="market",
    version="0.2.2",
    author="sajjad_shokrgozar",
    author_email="shokrgozarsajjad@gmail.com",
    description="A Python package for market analysis and pricing",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/sajjad-shokrgozar/market",
    packages=find_packages(),
    package_data={'market': ['firms_info.csv']},
    install_requires=[
        "pandas",
        "numpy",
        "scipy",
        "requests",
        "jdatetime",
        "helpers",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    license="MIT",
    license_files=["LICENSE"],  # ✅ Correct field
)
