from setuptools import setup, find_packages
import os
from pathlib import Path

this_directory = Path(__file__).parent

VERSION = '0.0.4'
DESCRIPTION = 'Representations used in COVID19 detection paper.'
LONG_DESCRIPTION = long_description = (this_directory /
                                       "README.md").read_text()

# Setting up
setup(name="c19rep",
      version=VERSION,
      author="Boshko Koloski",
      author_email="boshko.koloski@gmail.com",
      description=DESCRIPTION,
      long_description_content_type="text/markdown",
      long_description=LONG_DESCRIPTION,
      packages=find_packages(),
      install_requires=find_packages(),
      keywords=['python', 'covid19 fake news detection', 'nlp'],
      classifiers=[
          "Development Status :: 1 - Planning",
          "Intended Audience :: Developers",
          "Programming Language :: Python :: 3",
          "Operating System :: Unix",
      ])
