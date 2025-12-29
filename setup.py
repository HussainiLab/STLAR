import re
import pathlib
import setuptools

# Extract version from hfoGUI/__main__.py without importing heavy deps
root = pathlib.Path(__file__).parent
main_py = (root / 'hfoGUI' / '__main__.py').read_text(encoding='utf-8')
m = re.search(r"version\s*=\s*['\"]([^'\"]+)['\"]", main_py)
version = m.group(1) if m else '0.0.0'

with open("README.md", "r") as fh:
    long_description = fh.read()

pkgs = setuptools.find_packages()
print('found these packages:', pkgs)

pkg_name = "stlar"

setuptools.setup(
    name=pkg_name,
    version=version,
    author="Geoffrey Barrett",
    author_email="geoffrey.m.barrett@gmail.com",
    description="STLAR - unified spatio-temporal LFP analysis GUI.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/HussainiLab/STLAR",
    packages=pkgs,
    install_requires=
    [
        'PyQt5',
        'pillow',
        'numpy',
        'pyqtgraph',
        'scipy',
        'matplotlib',
        'pandas',
        'pyfftw'
    ],
    package_data={'stlar': ['img/*.png'], 'hfoGUI': ['img/*.png']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
)
