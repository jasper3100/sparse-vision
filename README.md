# An attempt to discover sparse feature circuits in vision models

This repository contains the code used for the experiments conducted by Jasper Toussaint for his master thesis at the University of Tübingen in 2024 titled "An attempt to discover sparse feature circuits in vision models". 

Most folder names are indicative of what they contain. Here are some additional elaborations:
- `supplementary_files_1` generally contains unused and old code
- `supplementary_files_2` contains used but supplementary code, f.e., to create figures.

### How to use this repository?
- set parameters in specify_parameters.py and run this file to generate a .txt file allowing to iterate over the parameter combinations
- locally: just run main.py (might be inconvenient for running several parameter combinations, as these are executed sequentially, but for one parameter combination it is fine); on the cluster: submit a job getting the parameters from the txt file and repeatedly executing main.py
- main.py calls execute_project.py which calls model_pipeline.py, which contains the core code

The filenames and locations in this codebase are hard coded and have to be adjusted for personal usage. Moreover, the code for computing the machine interpretability score (MIS) [[1]](#1) might have to be requested from the authors.

### References
<a id="1">[1]</a> 
Zimmermann, R. S., Klindt, D. A., and Brendel, W. (2024).
Measuring Mechanistic Interpretability at Scale Without Humans.
ICLR 2024 Workshop on Representational Alignment.