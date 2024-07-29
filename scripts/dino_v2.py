import os
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import torch
import numpy as np

def list_image_files(root_dir):
    """
    List all image files in the root directory and its subdirectories.
    """
    image_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, file))
    return image_files

def process_and_infer_images(image_files, chunk_size, processor, model, output_dir, max_size, device):
    """
    Process and perform inference on images in chunks. Save the last hidden states to the output directory.
    """
    for i in range(0, max_size, chunk_size):
        chunk_files = image_files[i:i+chunk_size]
        images = [Image.open(image_file) for image_file in chunk_files]
        inputs = processor(images=images, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state

        # Use class token only
        last_hidden_states = torch.stack([
            last_hidden_states[:, 0, :],
            torch.mean(last_hidden_states[:, 1:, :], dim=1)
        ], dim=1)

        for j, image_file in enumerate(chunk_files):
            base_name = os.path.basename(image_file)
            output_path = os.path.join(output_dir, base_name.split(".")[0] + '.npy')
            np.save(output_path, last_hidden_states[j].cpu().numpy())

        print(f"Processed chunk {i // chunk_size + 1}/{(max_size + chunk_size - 1) // chunk_size}")

if __name__ == '__main__':
    # Define the root directory containing subdirectories with images
    root_dir = './dataset/images'
    output_dir = './output'
    chunk_size = 256  # Define the chunk size

    # Create output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # List all images in the root directory and its subdirectories
    image_files = list_image_files(root_dir)

    # Load the image processor and model
    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small')
    model = AutoModel.from_pretrained('facebook/dinov2-small')

    # Move model to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Process and perform inference on images in chunks
    process_and_infer_images(image_files, chunk_size, processor, model, output_dir, len(image_files), device)
