import os
import torch

class ModelSaver:
    def __init__(self, model, folder_path):
        self.model = model
        self.folder_path = folder_path

    def save_model_weights(self, file_name='model_weights.pth'):
        # Ensure the folder exists; create it if it doesn't
        os.makedirs(self.folder_path, exist_ok=True)

        # Save model weights
        file_path = os.path.join(self.folder_path, file_name)
        torch.save(self.model.state_dict(), file_path)
        print(f"Successfully stored model weights in {file_path}")

# Example usage:
# Assuming you have a model (sae) and a folder path
# sae = ...  # Your model instance
# sae_weights_folder_path = ...  # Your folder path
# model_saver = ModelSaver(sae, sae_weights_folder_path)
# model_saver.save_model_weights('sae_weights.pth')
