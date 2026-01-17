"""
Training of DCGAN network on MNIST dataset with Discriminator
and Generator imported from models.py

Programmed by Aladdin Persson <aladdin.persson at hotmail dot com>
* 2020-11-01: Initial coding
* 2022-12-20: Small revision of code, checked that it works with latest PyTorch version
"""

import os
import sys
import glob
import shutil
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import Discriminator, Generator, initialize_weights

# Hyperparameters etc.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE_G = 2e-4  # could also use two lrs, one for generator and one for disccriminator
LEARNING_RATE_D = 2e-4
BATCH_SIZE = 128
IMAGE_SIZE = 64
CHANNELS_IMG = 1
NOISE_DIM = 100
NUM_EPOCHS = 3000
FEATURES_DISC = 64
FEATURES_GEN = 64

gen_losses = []
disc_losses = []
epoch_list = []


transforms = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]
        ),
    ]
)

class SafeImageFolder(Dataset):
    def __init__(self, root, transform=None, extensions=(".png", ".jpg", ".jpeg")):
        self.transform = transform
        self.images = []
        for dirpath, dirnames, filenames in os.walk(root):
            # Skip checkpoint folders
            if "checkpoints" in dirpath:
                continue
            for f in filenames:
                if f.lower().endswith(extensions):
                    self.images.append(os.path.join(dirpath, f))
        if not self.images:
            raise RuntimeError(f"No image files found in {root}")
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image, 0  # Dummy label for GANs (unsupervised)

def cleanup_checkpoints(checkpoint_folder):
    """Deletes all checkpoints except the best model if specified."""
    checkpoint_files = glob.glob(os.path.join(checkpoint_folder, "*.pth"))
    for ckpt in checkpoint_files:
        try:
            os.remove(ckpt)
            print(f"🧹 Deleted checkpoint: {ckpt}")
        except Exception as e:
            print(f"⚠️ Failed to delete {ckpt}: {e}")

def save_checkpoint(epoch, gen, disc, opt_gen, opt_disc, folder_path):
    os.makedirs(folder_path, exist_ok=True)
    checkpoint_path = os.path.join(folder_path, f"checkpoint_epoch_{epoch}.pth")
    torch.save({
        "epoch": epoch,
        "gen_state_dict": gen.state_dict(),
        "disc_state_dict": disc.state_dict(),
        "opt_gen_state_dict": opt_gen.state_dict(),
        "opt_disc_state_dict": opt_disc.state_dict(),
    }, checkpoint_path)
    print(f"\n\n💾 Checkpoint saved at {checkpoint_path}\n")


def load_latest_checkpoint(gen, disc, opt_gen, opt_disc, folder_path):
    if not os.path.exists(folder_path):
        print("\n\n⚠️ No checkpoint folder found.\n\n")
        return 0  # start from epoch 0
    checkpoints = [f for f in os.listdir(folder_path) if f.endswith(".pth")]
    if not checkpoints:
        print("\n\n⚠️ No checkpoints found.\n\n")
        return 0
    latest = sorted(checkpoints, key=lambda x: int(x.split("_")[2].split(".")[0]))[-1]
    checkpoint = torch.load(os.path.join(folder_path, latest))
    gen.load_state_dict(checkpoint["gen_state_dict"])
    disc.load_state_dict(checkpoint["disc_state_dict"])
    opt_gen.load_state_dict(checkpoint["opt_gen_state_dict"])
    opt_disc.load_state_dict(checkpoint["opt_disc_state_dict"])
    print(f"\n\n✅ Loaded checkpoint from {latest}\n\n")
    return checkpoint["epoch"] + 1  # resume from next epoch

def load_previous_checkpoint(gen, disc, opt_gen, opt_disc, folder_path):
    """Load the second-latest checkpoint (used for rollback)."""
    if not os.path.exists(folder_path):
        print("\n\n⚠️ No checkpoint folder found.\n\n")
        return 0  # start from epoch 0
    checkpoints = [f for f in os.listdir(folder_path) if f.endswith(".pth")]
    if not checkpoints:
        print("\n\n⚠️ No checkpoints found.\n\n")
        return 0
    checkpoints = [f for f in os.listdir(folder_path) if f.endswith(".pth")]
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("_")[2].split(".")[0]))
    if len(checkpoints) > 1:
        prev = checkpoints[-2]
    else:
        prev = checkpoints[-1]
    checkpoint = torch.load(os.path.join(folder_path, prev))
    gen.load_state_dict(checkpoint["gen_state_dict"])
    disc.load_state_dict(checkpoint["disc_state_dict"])
    opt_gen.load_state_dict(checkpoint["opt_gen_state_dict"])
    opt_disc.load_state_dict(checkpoint["opt_disc_state_dict"])
    print(f"\n⏪ Rolled back to {prev} due to divergence.\n")
    return checkpoint["epoch"] + 1

def train_dataset(data_dir, current_num, total_num):
    torch.cuda.empty_cache()
    
    dataset = SafeImageFolder(root=data_dir, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    gen = Generator(NOISE_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
    disc = Discriminator(CHANNELS_IMG, FEATURES_DISC).to(device)
    initialize_weights(gen)
    initialize_weights(disc)
    
    folder_path = "./GAN/logs"
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        shutil.rmtree(folder_path)
        print(f"Deleted folder: {folder_path}")
    else:
        print(f"Folder does not exist: {folder_path}")

    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE_G, betas=(0.5, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE_D, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    fixed_noise = torch.randn(32, NOISE_DIM, 1, 1).to(device)
    writer_real = SummaryWriter(f"{folder_path}/real")
    writer_fake = SummaryWriter(f"{folder_path}/fake")
    step = 0

    # Folder for checkpoints per Unicode
    checkpoint_folder = os.path.join(data_dir, "checkpoints")
    epoch = load_latest_checkpoint(gen, disc, opt_gen, opt_disc, checkpoint_folder)

    gen.train()
    disc.train()

    while epoch < NUM_EPOCHS:
        epoch_gen_loss = 0.0
        epoch_disc_loss = 0.0
        num_batches = 0
        # Target labels not needed! <3 unsupervised
        for batch_idx, (real, _) in enumerate(dataloader):
            real = real.to(device)
            noise = torch.randn(BATCH_SIZE, NOISE_DIM, 1, 1).to(device)
            fake = gen(noise)

            ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
            disc_real = disc(real).reshape(-1)
            loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
            disc_fake = disc(fake.detach()).reshape(-1)
            loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            loss_disc = (loss_disc_real + loss_disc_fake) / 2
            disc.zero_grad()
            loss_disc.backward()
            opt_disc.step()

            ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
            output = disc(fake).reshape(-1)
            loss_gen = criterion(output, torch.ones_like(output))
            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            # Check for divergence
            diff = abs(loss_disc.item() - loss_gen.item())
            if diff > 40:
                print(f"⚠️ Divergence detected  at epoch {epoch}, (Δ={diff:.2f}). Rolling back to previous checkpoint.")
                new_start = load_previous_checkpoint(gen, disc, opt_gen, opt_disc, checkpoint_folder)
                if new_start:
                    # epoch = new_start
                    break
                else:
                    print("⚠️ No previous checkpoint available, continuing...")
                continue

            epoch_gen_loss += loss_gen.item()
            epoch_disc_loss += loss_disc.item()
            num_batches += 1

            # Print losses occasionally and print to tensorboard
            if batch_idx % 100 == 0:
                msg = (
                    f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(dataloader)} "
                    f"Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f} "
                    f"Processing ({current_num}/{total_num}) folder "
                    f"Current folder: {os.path.basename(os.path.normpath(data_dir))}"
                )
                # Print in place (overwrite the same line)
                sys.stdout.write("\r" + msg)
                sys.stdout.flush()

                with torch.no_grad():
                    fake = gen(fixed_noise)
                    # take out (up to) 32 examples
                    img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                    img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)

                    writer_real.add_image("Real", img_grid_real, global_step=step)
                    writer_fake.add_image("Fake", img_grid_fake, global_step=step)

                step += 1
        
        # end of each epoch
        if (epoch + 1) % 500 == 0:
            save_checkpoint(epoch + 1, gen, disc, opt_gen, opt_disc, checkpoint_folder)

        if num_batches > 0:
            gen_losses.append(epoch_gen_loss / num_batches)
            disc_losses.append(epoch_disc_loss / num_batches)
            epoch_list.append(epoch)
        epoch += 1


    # ================================
    # After Training: Save Generator & Delete Excesses Checkpoints
    # ================================

    plt.figure(figsize=(8, 5))
    plt.plot(epoch_list, gen_losses, label="Generator Loss")
    plt.plot(epoch_list, disc_losses, label="Discriminator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("GAN Training Loss Curve")
    plt.legend()
    plt.grid(True)

    plot_path = os.path.join(data_dir, "gan_loss_curve.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"📈 GAN loss curve saved to {plot_path}")

    name = "generator_dcGAN.pth"
    torch.save(gen.state_dict(), os.path.join(data_dir, name))
    print("✅ Generator model saved to ", os.path.join(data_dir, name))
    cleanup_checkpoints(checkpoint_folder)
    print("\n\n")


## Test run

train_dataset("./unselected/U+4E08", 1, 1)