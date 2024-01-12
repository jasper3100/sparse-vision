# transforming_networks_to_sparse_basis

Contents of this repository: 

- main.py: main script to define parameters (and run the pipeline, NOT WORKING YET)

- resnet50.py: instantiate pre-trained ResNet50 model
- data.py: dataset
- sae.py: sparse autoencoder model
- sparse_loss.py: custom loss function for the sparse autoencoder
- aux.py: auxiliary functions: print classification results of model and print all layer names

- extract_intermediate_features.py: extract intermediate features of a specific layer of the model
- train_sae.py: train sparse autoencoder 
- evaluate_model_on_adjusted_features.py: output of intermediate layer --> autoencoder --> into model
