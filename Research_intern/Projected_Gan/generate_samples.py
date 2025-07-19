import torch
from torchvision.utils import save_image
import os
from generator import Generator
import glob

# Configs
nz = 100
image_size = 256
checkpoint_dir = "checkpoints"  # Directory containing all checkpoints
output_base_dir = "synthetic_samples"

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Find all generator checkpoints
checkpoint_paths = sorted(glob.glob(f"{checkpoint_dir}/*/Generator"))
print(f"Found {len(checkpoint_paths)} checkpoints")

# Create generator model
gen = Generator(nz=nz, im_size=image_size).to(device)

# Process each checkpoint
for checkpoint_path in checkpoint_paths:
    # Extract epoch number from path (assumes structure: checkpoints/epoch_X/Generator)
    epoch = checkpoint_path.split('/')[1]
    output_dir = os.path.join(output_base_dir, f"epoch_{epoch}")
    os.makedirs(output_dir, exist_ok=True)

    # Load weights
    gen.load_state_dict(torch.load(checkpoint_path, map_location=device))
    gen.eval()

    # Generate images
    num_images = 100  # Generate 100 synthetic images per checkpoint
    noise = torch.randn(num_images, nz, device=device)

    with torch.no_grad():
        fake_images = gen(noise)

    # Save all individual images
    for i in range(num_images):
        save_image(fake_images[i], os.path.join(output_dir, f"sample_{i+1}.png"), normalize=True)

    print(f"✅ Generated {num_images} samples for epoch {epoch} at {output_dir}")

print("Completed all checkpoint generations!")
