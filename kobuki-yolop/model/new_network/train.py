import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
from model import FullModelWithAPF  # Ensure this imports your model
from torchvision import transforms
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class CustomSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.images[idx])
        mask_name = self.images[idx].replace('.jpg', '.png')  # Adjust to your naming
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Assuming masks are grayscale (2D)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        # Ensure the mask is a 3D tensor: [height, width] -> [1, height, width]
        mask = mask.squeeze(0)  # Remove extra channel dimension if present

        return image, mask


# Directory paths
image_dir = "/home/naren/ICRA_work/dataset/dataset/images/"
mask_dir = "/home/naren/ICRA_work/dataset/dataset/da_seg_annotations/"

# Define a training loop with multiple supervisors
import torch.nn.functional as F

# Define a simplified training loop with final output loss
import torch.nn.functional as F

# Define a simplified training loop with final output loss
def train_model(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for data in tqdm(dataloader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        # Forward pass - only use sup1 as the final output
        sup1, _, _, _ = model(inputs)

        # Upsample the output to match the size of the ground truth labels
        sup1 = F.interpolate(sup1, size=(labels.shape[1], labels.shape[2]), mode='bilinear', align_corners=False)

        # Convert labels to Long type for CrossEntropyLoss
        labels = labels.long()

        # Compute loss only for the final output
        loss = criterion(sup1, labels)

        loss.backward()  # Backpropagation
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(dataloader)
    return epoch_loss


# Main training function
import os

def main():
    # Create directory if it doesn't exist
    save_dir = "/home/naren/ICRA_work/models/"
    os.makedirs(save_dir, exist_ok=True)

    # Hyperparameters
    epochs = 50  # You can adjust this
    learning_rate = 0.0001
    batch_size = 1  # As per your requirement

    # Load the dataset
    transform = transforms.Compose([
        transforms.Resize((640, 384)),
        transforms.ToTensor()
    ])
    train_dataset = CustomSegmentationDataset(image_dir, mask_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize the model, loss function, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FullModelWithAPF().to(device)
    criterion = nn.CrossEntropyLoss()  # You can try adding DiceLoss later
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # To save the best model
    best_loss = float('inf')
    best_epoch = 0

    # Training loop
    for epoch in range(epochs):
        epoch_loss = train_model(model, train_loader, optimizer, criterion, device)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}")

        # Save the model for every epoch
        torch.save(model.state_dict(), os.path.join(save_dir, f"model_epoch_{epoch + 1}.pth"))

        # Check if this is the best model based on loss
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_epoch = epoch + 1
            # Save the best model
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
            print(f"Best model saved at epoch {best_epoch} with loss {best_loss:.4f}")

    print(f"Training complete. Best model at epoch {best_epoch} with loss {best_loss:.4f}")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()

