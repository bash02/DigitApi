from torchvision.datasets import EMNIST
import torchvision.transforms as transforms
from PIL import Image
import os

# Set directory for extracted images
output_dir = './alphabet'
os.makedirs(output_dir, exist_ok=True)

# Load EMNIST dataset (letters split) from local files (disable download)
dataset = EMNIST(root='./alphabet', split='letters', train=True, download=False)

# Extract and save images
for idx, (img, label) in enumerate(dataset):
    letter = chr(label + 64)  # Convert label to letter (1 -> A, 2 -> B, ..., 26 -> Z)
    letter_dir = os.path.join(output_dir, letter)
    os.makedirs(letter_dir, exist_ok=True)

    img = img.convert("L")  # Ensure grayscale format
    img.save(os.path.join(letter_dir, f'{letter}_{idx}.png'))

    if idx % 1000 == 0:
        print(f"Extracted {idx} images")
