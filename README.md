# Transforming networks to sparse basis

Overview:
- set parameters in specify_parameters.py and run this file to generate a txt file allowing to iterate over the parameter combinations
- locally: just run main.py (might be inconvenient for running several parameter combinations, as these are executed sequentially, but for one parameter combination it is fine); on the cluster: submit a job getting the parameters from the txt file and repeatedly executing main.py
- main.py calls execute_project.py which calls model_pipeline.py
- model_pipeline.py contains the core code
