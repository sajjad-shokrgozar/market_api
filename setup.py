from setuptools import setup, find_packages

setup(
    name="market_api",
    version="0.1.4",
    author="sajjad_shokrgozar",
    author_email="shokrgozarsajjad@gmail.com",
    description="A Python package for market analysis and option pricing",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/sajjad-shokrgozar/market_api",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "scipy",
        "requests",
        "matplotlib",
        "jdatetime"
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
