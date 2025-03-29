import os
from PIL import Image, ImageDraw, ImageFont

# Create a folder to save images
output_dir = "alphabet_images"
os.makedirs(output_dir, exist_ok=True)

# Fonts (you can add more `.ttf` fonts to make it diverse)
font_paths = ["arial.ttf", "times.ttf", "DejaVuSans-Bold.ttf"]

# Text types: A-Z, a-z, sample words
characters = [chr(c) for c in range(65, 91)] + [chr(c) for c in range(97, 123)]
sample_words = ["Hello", "World", "Python", "OCR", "Vision"]

for font_path in font_paths:
    try:
        font = ImageFont.truetype(font_path, size=48)
    except:
        continue
    for text in characters + sample_words:
        img = Image.new('L', (100, 100), color=255)
        draw = ImageDraw.Draw(img)
        w, h = draw.textsize(text, font=font)
        draw.text(((100 - w) / 2, (100 - h) / 2), text, font=font, fill=0)
        fname = f"{text}_{font_path.replace('.ttf','')}.png".replace(" ", "_")
        img.save(os.path.join(output_dir, fname))
