import torch
from torchvision import transforms, datasets
from torchvision.utils import save_image
from PIL import Image
import os


def load_data(data_path, batch_size):
    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    train_data = datasets.ImageFolder(data_path, transform=train_transforms)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    return train_data, trainloader


def save_transformed_images(dataset, save_dir, num_images=100):
    os.makedirs(save_dir, exist_ok=True)
    for idx, (image_tensor, label) in enumerate(dataset):
        if idx >= num_images:
            break
        # Denormalize before saving
        image_tensor = image_tensor * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_tensor = image_tensor + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        image_tensor = torch.clamp(image_tensor, 0, 1)  # Clamp to valid range
        save_path = os.path.join(save_dir, f"img_{idx}_label_{label}.png")
        save_image(image_tensor, save_path)


if __name__ == '__main__':
    dataset, loader = load_data("organized_skin_lesions", 32)
    save_transformed_images(dataset, "resized_images", num_images=100)
