# An attempt to discover sparse feature circuits in vision models

This repository contains the code used for the experiments conducted by Jasper Toussaint for his master thesis at the University of Tübingen in 2024 titled "An attempt to discover sparse feature circuits in vision models". 

Most folder names are indicative of what they contain. Here are some additional elaborations:
- `supplementary_files_1` generally contains unused and old code.
- `supplementary_files_2` contains used but supplementary code, f.e., to create figures.
- `model_pipeline.py` contains the core code. We train sparse autoencoders (SAEs) following [[1]](#1).
- `main.py` calls `execute_project.py`, which calls `model_pipeline.py`.
- `compute_ie.py` is based on the work by [[2]](#2) but adjusted to our code.

### How to use this repository?
- Set parameters in `specify_parameters.py` and run this file to generate a `.txt` file allowing to iterate over the parameter combinations.
- _when using the code locally_: Just run `main.py`. Running the code locally might be inconvenient with several parameter combinations as the code is executed sequentially for each combination.
- _when using the code on a computing cluster_: Submit a job getting the parameters from the `.txt` file and repeatedly executing `main.py`.
- The filenames and locations in this codebase are hard coded and have to be adjusted for personal usage. 
- Moreover, for using the code for computing the machine interpretability score (MIS) [[3]](#3) please ask the authors. It is not included here because at the time of writing it was not publicly accessible.

### References
<a id="1">[1]</a> 
Bricken, T., Templeton, A., Batson, J., Chen, B., Jermyn, A., Conerly, T., Turner, N.,
Anil, C., Denison, C., Askell, A., Lasenby, R., Wu, Y., Kravec, S., Schiefer, N., Maxwell,
T., Joseph, N., Hatfield-Dodds, Z., Tamkin, A., Nguyen, K., McLean, B., Burke, J. E.,
Hume, T., Carter, S., Henighan, T., and Olah, C. (2023). 
Towards monosemanticity: Decomposing language models with dictionary learning. 
Transformer Circuits Thread.

<a id="2">[2]</a> 
Marks, S., Rager, C., Michaud, E. J., Belinkov, Y., Bau, D., and Mueller, A. (2024). 
Sparse feature circuits: Discovering and editing interpretable causal graphs in language models.
arXiv preprint arXiv:2403.19647.

<a id="3">[3]</a> 
Zimmermann, R. S., Klindt, D. A., and Brendel, W. (2024).
Measuring Mechanistic Interpretability at Scale Without Humans.
ICLR 2024 Workshop on Representational Alignment.