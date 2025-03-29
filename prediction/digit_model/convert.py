import os
from torchvision import datasets

# Define dataset directory
mnist_root = "./digit"  # Path where MNIST is already stored
output_dir = "./mnist_extracted"  # Directory to save extracted images

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load MNIST dataset (assuming it's already downloaded)
mnist_data = datasets.MNIST(root=mnist_root, train=False, download=False)

# Loop through dataset and save images
for idx, (image, label) in enumerate(mnist_data):
    label_dir = os.path.join(output_dir, str(label))
    os.makedirs(label_dir, exist_ok=True)  # Create folder for each digit (0-9)

    image_path = os.path.join(label_dir, f"{idx}.png")
    image.save(image_path)  # Directly save without conversion

    if idx % 5000 == 0:  # Print progress every 5000 images
        print(f"Saved {idx} images...")

print("MNIST images extracted and saved successfully.")
