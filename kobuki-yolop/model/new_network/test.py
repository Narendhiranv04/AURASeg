import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import time
import os
import numpy as np
from model import FullModelWithAPF  # Your model
from torch.utils.data import Dataset

import os
from PIL import Image
import matplotlib.pyplot as plt

# Function to save visualizations
def save_output(inputs, labels, predictions, idx, output_dir):
    """Save input image, ground truth label, and predicted output to the runs folder."""
    
    # Convert tensors to numpy for saving
    inputs_np = inputs.cpu().squeeze().permute(1, 2, 0).numpy()  # Convert input to numpy
    labels_np = labels.cpu().squeeze().numpy()  # Convert labels to numpy
    predictions_np = predictions.argmax(dim=1).cpu().squeeze().numpy()  # Convert predictions to numpy
    
    # Convert input image to PIL and save
    input_image = Image.fromarray((inputs_np * 255).astype('uint8'))
    input_image.save(os.path.join(output_dir, f"input_{idx}.png"))

    # Save ground truth and predicted mask using matplotlib
    plt.imsave(os.path.join(output_dir, f"label_{idx}.png"), labels_np, cmap='gray')
    plt.imsave(os.path.join(output_dir, f"prediction_{idx}.png"), predictions_np, cmap='gray')



# IoU Calculation
def calculate_iou(pred, target, num_classes):
    ious = []
    pred = pred.argmax(dim=1)  # Get the predicted class from sup1

    # Debugging: Print shapes before IoU calculation
    print(f"pred shape: {pred.shape}, target shape: {target.shape}")

    for cls in range(num_classes):
        pred_cls = (pred == cls).float()  # Binary mask for class `cls`
        target_cls = (target == cls).float()  # Binary mask for class `cls` in ground truth

        # Ensure pred_cls and target_cls are the same size before calculating IoU
        intersection = (pred_cls * target_cls).sum().item()  # Intersection
        union = (pred_cls + target_cls).clamp(0, 1).sum().item()  # Union

        if union == 0:
            ious.append(float('nan'))  # No ground truth for this class
        else:
            ious.append(intersection / union)  # IoU = intersection / union

    return np.nanmean(ious)




class CustomSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            image_path = os.path.join(self.image_dir, self.images[idx])
            mask_name = self.images[idx].replace('.jpg', '.png')  # Adjust to your mask naming
            mask_path = os.path.join(self.mask_dir, mask_name)

            # Load image and mask
            image = Image.open(image_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")  # Assuming masks are grayscale (2D)

            if self.transform:
                image = self.transform(image)
                mask = self.transform(mask)

            return image, mask

        except Exception as e:
            print(f"Error loading image or mask at index {idx}: {e}")
            return None, None  # Return None if there's an error




# FPS Calculation
def calculate_fps(start_time, end_time, num_samples):
    return num_samples / (end_time - start_time)

# Test the model and compute FPS, IoU
def test_model(model, dataloader, device, num_classes=2):
    model.eval()
    total_iou = 0.0
    total_samples = 0
    start_time = time.time()

    # Create directory for saving results
    output_dir = "runs"
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(dataloader):
            if inputs is None or labels is None:
                continue  # Skip invalid samples

            inputs, labels = inputs.to(device), labels.to(device)
            total_samples += inputs.size(0)

            # Forward pass
            sup1, _, _, _ = model(inputs)

            # Resize the prediction to match the ground truth
            sup1 = F.interpolate(sup1, size=(labels.shape[-2], labels.shape[-1]), mode='bilinear', align_corners=False)

            # Compute IoU
            iou = calculate_iou(sup1, labels, num_classes)
            total_iou += iou

            # Save the input, ground truth, and predicted mask
            save_output(inputs, labels, sup1, idx, output_dir)

    end_time = time.time()
    avg_iou = total_iou / len(dataloader)
    fps = calculate_fps(start_time, end_time, total_samples)

    return avg_iou, fps




# Main testing function
def main():
    # Paths
    model_path = "/home/naren/ICRA_work/models/model_epoch_5.pth"  # Path to your trained model
    test_image_dir = "/home/naren/ICRA_work/dataset/dataset/images/test"
    test_mask_dir = "/home/naren/ICRA_work/dataset/dataset/da_seg_annotations/test"

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the dataset
    transform = transforms.Compose([
        transforms.Resize((640, 384)),
        transforms.ToTensor()
    ])
    
    test_dataset = CustomSegmentationDataset(test_image_dir, test_mask_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Load the model
    model = FullModelWithAPF().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Run testing
    avg_iou, fps = test_model(model, test_loader, device, num_classes=2)  # Adjust `num_classes` as needed

    print(f"Average IoU: {avg_iou:.4f}")
    print(f"FPS: {fps:.2f}")

if __name__ == "__main__":
    main()

