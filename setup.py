"""Setup configuration for the Weekly Trading Advisor package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="weekly-trading-advisor",
    version="0.1.0",
    author="Aaron",
    author_email="aaron@example.com",
    description="A tool for generating trading advice based on technical indicators",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/solvmanifold/LatentTrader",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "yfinance>=0.2.36",
        "ta>=0.10.0",
        "typer>=0.9.0",
        "rich>=13.0.0",
        "requests",
        "beautifulsoup4",
    ],
    entry_points={
        "console_scripts": [
            "weekly-trading-advisor=weekly_trading_advisor.cli:app",
        ],
    },
) 