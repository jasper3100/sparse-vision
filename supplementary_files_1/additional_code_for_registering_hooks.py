# ADDITIONAL CODE FOR REGISTERING HOOKS ON LAYERS

# Register hooks for each block within layer1
target_layer_name = 'layer1'
target_layer = None
for name, layer in model.named_modules():
    #print(name, layer)
    if name == target_layer_name:
        target_layer = layer
        break
#print(target_layer)

if target_layer is not None:
    # attach hook to first block of target_layer
    block_idx = 0
    block = target_layer[block_idx]
    block.register_forward_hook(lambda module, inp, out, idx=block_idx: hook(module, inp, out, f'{target_layer_name}_{idx}'))
    #for block_idx, block in enumerate(target_layer):
        #print(block_idx, block)
        #for layer_idx, layer in enumerate(block):
        #    print(layer_idx, layer)

        #block.register_forward_hook(lambda module, inp, out, idx=block_idx: hook(module, inp, out, f'{target_layer_name}_{idx}'))
        # the last part in the above line means that the blocks in layer1 will be named as layer1_0, layer1_1, layer1_2, layer1_3
        # those blocks consist of several layers