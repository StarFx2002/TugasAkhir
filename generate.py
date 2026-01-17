import torch
import torchvision
from model import Generator

# Hyperparameters etc.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHANNELS_IMG = 1
NOISE_DIM = 100
FEATURES_GEN = 64

gen = Generator(NOISE_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)


# Generate multiple single images after training
gen.load_state_dict(torch.load("./unselected/U+4E08/generator_dcGAN.pth"))  # if you saved checkpoint
gen.eval()
with torch.no_grad():
    num_images = 3  # how many images you want
    noise = torch.randn(num_images, NOISE_DIM, 1, 1).to(device)
    fake_images = gen(noise)

    for i in range(num_images):
        torchvision.utils.save_image(
            fake_images[i], f"generated_{i+1}.png", normalize=True
        )

print(f"🎨 {num_images} images saved as generated_1.png ... generated_{num_images}.png")