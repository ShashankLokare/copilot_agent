# Setup file for trading_agent package

from setuptools import setup, find_packages

setup(
    name="trading_agent",
    version="0.1.0",
    description="Production-grade multi-agent algorithmic trading system",
    author="Trading Systems Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "pytz>=2021.1",
        "pyyaml>=5.4.0",
        "scikit-learn>=1.0.0",
    ],
    extras_require={
        "ml": ["xgboost>=1.5.0"],
        "viz": ["matplotlib>=3.4.0", "jupyter>=1.0.0"],
        "brokers": [
            # "alpaca-trade-api>=0.60.0",
            # "nsepy>=0.1",
            # "ccxt>=1.60.0",
        ],
        "dev": ["pytest>=6.0", "black", "mypy", "flake8"],
    },
)
