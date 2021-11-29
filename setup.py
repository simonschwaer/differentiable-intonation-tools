from setuptools import setup, find_packages
from dit.version import version as dit_version

setup(
    name             = "dit",
    version          = dit_version,
    description      = "Differentiable Intonation Tools (dit)",
    author           = "Simon Schwär, Sebastian Rosenzweig, Meinard Müller",
    author_email     = "mail@simon-schwaer.de",
    packages         = find_packages(),
    license          = "MIT",
    install_requires = [
        "librosa >= 0.8.0, < 1.0.0",
        "libtsm >= 1.1.0, < 2.0.0",
        "numpy >= 1.17.0, < 2.0.0",
        "scipy >= 1.7.0, < 2.0.0",
    ],
    python_requires  = ">=3.7, <4.0",
    extras_require   = {
        "tests": ["pytest == 6.2.*"],
    }
)