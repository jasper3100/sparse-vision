# Transforming networks to sparse basis

For intalling subdirectories as Python packages, which is necessary for imports from the parent directory to work, do the following:

On Linux/Mac, run: 

`#!/bin/bash
pip install -e .`

These 2 lines can also be put into an `install.sh` file, which can be run from the command line. Before that, this script needs to be made executable: `chmod +x install.sh`

On Windows, run:

`@echo off
pip install -e .`

Alternatively, one can run the `install.bat` file, which contains those 2 lines: `.\install.bat` 