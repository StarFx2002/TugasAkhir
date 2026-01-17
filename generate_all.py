import os
import torch
import torchvision
from model import Generator

root_dir = "./unselected"

# Hyperparameters etc.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHANNELS_IMG = 1
NOISE_DIM = 100
FEATURES_GEN = 64

generator_name = "generator_dcGAN.pth"

i = 0
for unicode in sorted(os.listdir(root_dir)):
    unicode_path = os.path.join(root_dir, unicode)

    if os.path.isdir(unicode_path):
        generatorFound = False
        for files in os.listdir(unicode_path):
            if files == generator_name:
                generatorFound = True
                break
        if not generatorFound:
            print(f"Skipping folder generator_name not found: {unicode}")
            continue
    
    generator_path = os.path.join(unicode_path, generator_name)
    image_path = os.path.join(unicode_path, "images")

    num_images = 250 - len(os.listdir(image_path))

    if num_images <= 0:
        print(f"Skipping folder (already has 250 images): {unicode}")
        continue

    gen = Generator(NOISE_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
    gen.load_state_dict(torch.load(generator_path))
    gen.eval()

    with torch.no_grad():
        noise = torch.randn(num_images, NOISE_DIM, 1, 1).to(device)
        fake_images = gen(noise)

        for i in range(num_images):
            torchvision.utils.save_image(
                fake_images[i], f"{image_path}\\generated_{i+1}.png", normalize=True
            )

    print(f"Generated {num_images} images in {unicode} folder.")