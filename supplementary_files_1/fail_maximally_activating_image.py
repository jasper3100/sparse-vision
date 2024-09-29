# WAY TOO COMPLICATED AND UNSUCCESSFUL WAY OF CREATING A MAXIMALLY ACTIVATING IMAGE FOR ONE NEURON
# IN AN MLP FOR MNIST IMAGE CLASSIFICATION


def act_update(act, neuron_idx, artificial_image):
    act = act.squeeze()
    # consider the output of the neuron we're interested in
    act = act[neuron_idx]
    # since we want to maximize the activation value of this neuron, we minimize the negative of the activation value
    loss = -act
    #optimizer.zero_grad()
    loss.backward()
    # we update the input image
    #optimizer.step()
    # clamp the image values to the valid range
    #artificial_image.data.clamp_(0, 1)
    # Update only 60 pixel values
    with torch.no_grad():
        current_white_pixels = torch.sum(artificial_image.data == 1.0)
        
        print(current_white_pixels)
        if current_white_pixels <= 30: 
            # We update the 60 pixels with the highest gradient magnitude
            # Sort pixels by their gradient magnitude
            print("Update")
            grad_magnitude = -1*artificial_image.grad.abs()
            _, indices = grad_magnitude.view(-1).topk(80)
            
            for index in indices:
                gradient = artificial_image.grad[0, index]
                surrounding_indices = []
                row, col = divmod(index.item(), 28)
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        new_row = row + i
                        new_col = col + j
                        if 0 <= new_row < 28 and 0 <= new_col < 28:
                            surrounding_index = new_row * 28 + new_col
                            # we add a bit of the gradient of the surrounding pixels
                            gradient += 0.5*artificial_image.grad[0, surrounding_index]
                            surrounding_indices.append(surrounding_index)
                # get the gradients of the surrounding pixels
                surrounding_grads = artificial_image.grad[0, surrounding_indices]
                # all of the surrounding gradients should have the same sign as the gradient of the pixel
                # we want to update, otherwise this pixel might be an outlier --> we don't update it
                if torch.sum(surrounding_grads.sign() == artificial_image.grad[0, index].sign()) >= len(surrounding_indices)*0.8:
                    # we add a portion of the surrounding 
                    artificial_image.data[0, index] -= 0.01 * gradient

            # get the indices which are not in "indices"
            #remaining_indices = torch.tensor([i for i in range(28*28) if i not in indices])
            # update the remaining_indices by a tiny amount
            #artificial_image.data[0, remaining_indices] -= 0.01 * artificial_image.grad[0, remaining_indices]

            # Update only those pixel values
            #artificial_image_before = artificial_image.data.clone()
            #artificial_image.data[0, indices] -= 0.5 * artificial_image.grad[0, indices]
            # Clamp values to [0, 1]
            # check if the image is the same as before
            #print(torch.all(artificial_image_before == artificial_image.data))
            MNIST_mean = 0.1307
            MNIST_std = 0.3081
            artificial_image = (artificial_image - MNIST_mean) / MNIST_std
            artificial_image.data.clamp_(0, 1)


        else: 
            indices = None
    return artificial_image