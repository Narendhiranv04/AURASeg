import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for visualization
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import os
from model import FullModelWithAPF

# Preprocess input image
def preprocess_image(image_path, device):
    # Load the image and apply transformations
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((640, 384)),  # Resizing to the input size of your model
        transforms.ToTensor()
    ])
    img = transform(img).unsqueeze(0).to(device)  # Add batch dimension
    return img

# Visualize output segmentation mask
def visualize_output(output):
    output = output.squeeze(0).argmax(0).cpu().detach().numpy()
    plt.imshow(output, cmap='gray')
    plt.show()

# Main inference function
def main():
    # Path to the test image
    image_path = "/home/naren/ICRA_work/inference_2.jpg"  # Specify the path to your test image
    model_path = "/home/naren/ICRA_work/models/model_epoch_5.pth"  # Path to your saved best model

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    model = FullModelWithAPF().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set model to evaluation mode

    # Preprocess the input image
    img = preprocess_image(image_path, device)

    # Run inference
    with torch.no_grad():
        sup1, _, _, _ = model(img)  # Only take the final output (sup1)

        # Upsample the output to the original size of the image
        sup1 = F.interpolate(sup1, size=(img.shape[2], img.shape[3]), mode='bilinear', align_corners=False)

        # Visualize the output
        visualize_output(sup1)

if __name__ == "__main__":
    main()

