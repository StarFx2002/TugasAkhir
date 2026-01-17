import os
import sys
import subprocess

root_dir = "./unselected"
generator_name = "generator_dcGAN.pth"

i = 0
for unicode in sorted(os.listdir(root_dir)):
    i += 1
    unicode_path = os.path.join(root_dir, unicode)

    if os.path.isdir(unicode_path):
        generatorFound = False
        for files in os.listdir(unicode_path):
            if files == generator_name:
                generatorFound = True
                break
        if generatorFound:
            print(f"Skipping folder {generator_name}: {unicode}")
            continue
        print(f"Processing folder: {unicode}\n\n")
        # Run train_dataset in a fresh Python process
    subprocess.run([
        sys.executable, "train_runner.py",
        unicode_path, str(i), str(len(os.listdir(root_dir)))
    ], check=True)