# Transforming networks to sparse basis

First, set the parameters in `main.py`. Then, execute in this order:
- `extract_intermediate_features.py`: extract intermediate features of a specific layer of the model
- `train_sae.py`: train sparse autoencoder 
- `evaluate_model_on_adjusted_features.py`: output of intermediate layer --> autoencoder --> into model
- `evaluation_metrics.py`: evaluate whether the adjusted model is close to the original model

If the autoencoder is already trained, then one can of course run `evaluate_model_on_adjusted_features.py` without running the other files.

Contents of this repository: 

- `main.py`: main script to define parameters (and run the pipeline, NOT WORKING YET)
- `model.py`: instantiate the model to be used
- `data.py`: dataset
- `sae.py`: sparse autoencoder model
- `sparse_loss.py`: custom loss function for the sparse autoencoder
- `auxiliary_functions.py`: auxiliary functions: print classification results of model and print all layer names
- `extract_intermediate_features.py`: extract intermediate features of a specific layer of the model
- `train_sae.py`: train sparse autoencoder 
- `evaluate_model_on_adjusted_features.py`: output of intermediate layer --> autoencoder --> into model
- `evaluation_metrics.py`: evaluate whether the adjusted model is close to the original model
