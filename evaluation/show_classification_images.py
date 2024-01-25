import matplotlib.pyplot as plt
import numpy as np

def show_classification_with_images(input_images, target_ids, class_names, scores, predicted_classes, scores_2=None, predicted_classes_2=None):
    number_of_images = 10  # show only the first n images, 
    # for showing all images in the batch use len(predicted_classes)
    fig, axes = plt.subplots(1, number_of_images + 1, figsize=(20, 3))

    # Add a title column to the left
    title_column = 'True\nOriginal Prediction\nModified Prediction'
    axes[0].text(0.5, 0.5, title_column, va='center', ha='center', fontsize=8, wrap=True)
    axes[0].axis('off')

    for i in range(number_of_images):
        # Unnormalize the image
        img = input_images[i] / 2 + 0.5
        npimg = img.numpy()

        # Display the image with true label, predicted label, and score
        axes[i + 1].imshow(np.transpose(npimg, (1, 2, 0)))

        if scores_2 is not None:  # mode == 'compare_models':
            title = f'{class_names[target_ids[i]]}\n{predicted_classes[i]} ({scores[i].item():.1f}%)\n{predicted_classes_2[i]} ({scores_2[i].item():.1f}%)'
        else:
            title = f'{class_names[target_ids[i]]}\n{predicted_classes[i]} ({scores[i].item():.1f}%)'

        axes[i + 1].set_title(title, fontsize=8)
        axes[i + 1].axis('off')

    plt.subplots_adjust(wspace=0.5)  # Adjust space between images
    plt.show()