from utils import *

# SPECIFY DESIRED MODEL HERE
model_name = 'inceptionv1'

model = load_model(model_name)#, 32, 10)

# print all layers
#for name, module in model.named_modules():
#    print(name)
# All "." in the name refer to submodules

print("-------------------")

# print only the parent modules
i = 0
for name, module in model.named_children():
    print(name)
    i += 1
print(i)

# I just train my first SAE on the following layer layer1.0.conv1

m = getattr(model, "mixed3b_3x3_pre_relu_conv")
print(m)


'''
layer_name = 'layer1'  

layer = getattr(model, layer_name, None)

if layer is not None:
    print(layer)
else:
    print(f"Layer '{layer_name}' not found in the model.")

sublayer = getattr(layer, '0', None)

print(sublayer)

subsublayer = getattr(sublayer, 'conv1', None)

print(subsublayer)
'''