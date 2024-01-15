# ATTEMPT TO BUILD A NEW MODEL STARTING FROM INTERMEDIATE LAYER

# NOTE: THIS DOES NOT WORK IF THE MODEL IS NOT A SEQUENTIAL MODEL. 
# F.E. RESNET50 IS NOT A SEQUENTIAL MODEL. IT HAS A FLATTEN OPERATION BETWEEN
# LAYERS AND SKIP CONNECTIONS. 

# This is a code snippet that requires other code from the repository to run. 
# It is not a standalone code and only demonstrates some basic functionality, which 
# could be used for reference at a later point.

'''
Suppose we extracted the feature maps from the intermediate layer with index i. 
Hence, we want to build a new model starting from layer i+1.
'''
def build_new_model(modules, intermediate_layer_index):
    new_model = nn.Sequential(*modules[(intermediate_layer_index + 1):-1])
    #newmodel = torch.nn.Sequential(*(list(model.children())[:-1]))
    return new_model

new_model = build_new_model(modules, layer_index) 
# modules is a list of all layers of the model
#print(new_model)
#print(new_model(activation))